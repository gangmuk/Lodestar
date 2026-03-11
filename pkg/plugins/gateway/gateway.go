/*
Copyright 2024 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/vllm-project/aibrix/pkg/cache"
	routing "github.com/vllm-project/aibrix/pkg/plugins/gateway/algorithms"
	"github.com/vllm-project/aibrix/pkg/plugins/gateway/ratelimiter"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
)

type RequestTiming struct {
	startTime            time.Time // When the request began processing
	firstTokenTime       time.Time // When the first token was received
	firstDecodeTokenTime time.Time // When the first decode token was received
	lastTokenTime        time.Time // When the last token was received
	totalTokenCount      int64
	prefillTokenCount    int64
	decodeTokenCount     int64 // Count of tokens in the decode phase
	IsPrefill            bool
}

type Server struct {
	redisClient         *redis.Client
	ratelimiter         ratelimiter.RateLimiter
	client              kubernetes.Interface
	cache               cache.Cache
	requestBuffers      sync.Map // Thread-safe map to track buffers per request
	streamingUsageCache sync.Map // Map to store usage information from streaming responses
	statusCode          sync.Map // Map to track status codes per request: requestID -> statusCode
	selectedPodIP       sync.Map // Map to track target pod per request: requestID -> podIP
	routingContexts     sync.Map // Map to store routing contexts for each request: requestID -> *types.RoutingContext
	requestHeaders      sync.Map // Map to cache request headers per request: requestID -> map[string]string
}

func NewServer(redisClient *redis.Client, client kubernetes.Interface) *Server {
	c, err := cache.Get()
	if err != nil {
		panic(err)
	}
	r := ratelimiter.NewRedisAccountRateLimiter("aibrix", redisClient, 1*time.Minute)

	// Initialize the routers
	routing.Init()

	server := &Server{
		redisClient: redisClient,
		ratelimiter: r,
		client:      client,
		cache:       c,
	}
	return server
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	var user utils.User
	var rpm, traceTerm int64
	var respErrorCode int
	var model string
	var routingAlgorithm types.RoutingAlgorithm
	var routerCtx *types.RoutingContext
	var stream, isRespError bool
	ctx := srv.Context()
	completed := false
	requestCounted := false
	inflightCleaned := false
	requestID := ""

	// Ensure exactly-once cleanup on any exit path (context cancel, EOF, recv error, send error).
	//
	// Inflight counters: cleanupInflightCounters is idempotent — it uses LoadAndDelete on
	// RequestTimings, so if calculateTimingMetrics already ran and deleted the timing entry
	// (normal completion), the call is a no-op. It also requires HasRouted() to avoid
	// underflow when routing failed in HandleRequestBody (inflight never incremented).
	//
	// Request counting: If AddRequestCount was called but DoneRequestTrace was not yet called
	// (requestCounted is still true), call DoneRequestCount to balance the counter.
	defer func() {
		if routerCtx != nil {
			if routerCtx.HasRouted() {
				if !inflightCleaned {
					s.cleanupInflightCounters(routerCtx, routerCtx.RequestID)
					s.cleanupPerRequestState(routerCtx, routerCtx.RequestID)
				}
			} else {
				// Routing failed — inflight counters were never incremented, but
				// per-request state (RequestTimings, tokens, etc.) may have been
				// stored before routing. Clean those up to prevent memory leaks.
				s.cleanupPerRequestState(routerCtx, routerCtx.RequestID)
			}
		}
		if requestCounted && routerCtx != nil {
			s.cache.DoneRequestCount(routerCtx, routerCtx.RequestID, model, traceTerm)
		}
		if routerCtx != nil {
			routerCtx.Delete()
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		resp := &extProcPb.ProcessingResponse{}
		switch v := req.Request.(type) {

		case *extProcPb.ProcessingRequest_RequestHeaders:
			requestID = getRequestID(v.RequestHeaders.Headers.Headers)
			klog.V(5).Infof("Before HandleRequestHeaders, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())
			resp, user, rpm, routingAlgorithm = s.HandleRequestHeaders(ctx, requestID, req)

		case *extProcPb.ProcessingRequest_RequestBody:
			klog.V(5).Infof("Before HandleRequestBody, requestID: %s, ctx.Err(): %v", requestID, ctx.Err())
			resp, model, routerCtx, stream, traceTerm = s.HandleRequestBody(ctx, requestID, req, user, routingAlgorithm)
			if routerCtx != nil {
				if routerCtx.Err() != nil {
					klog.ErrorS(routerCtx.Err(), "Routing context already canceled, using original",
						"requestID", routerCtx.RequestID)
				}
				ctx = routerCtx
				// Only set requestCounted if AddRequestCount was actually called.
				// HandleRequestBody calls AddRequestCount (line 244) only after routing
				// succeeds. If routing failed (targetPodIP=="" or getRequestMessage error),
				// it returns early with a non-nil routerCtx but AddRequestCount was never
				// called, so DoneRequestCount in the defer would underflow.
				if routerCtx.HasRouted() {
					requestCounted = true
				}
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			klog.V(5).Infof("Before HandleResponseHeaders, requestID: %s, ctx.Err(): %v", routerCtx.RequestID, ctx.Err())
			resp, isRespError, respErrorCode = s.HandleResponseHeaders(ctx, routerCtx.RequestID, model, req)
			if isRespError {
				klog.Errorf("Response headers processing error %d, requestID: %s, selectedPod: %s, model: %s", respErrorCode, routerCtx.RequestID, routerCtx.TargetAddress(), model)
			}
		case *extProcPb.ProcessingRequest_ResponseBody:
			respBody := req.Request.(*extProcPb.ProcessingRequest_ResponseBody)
			if isRespError {
				klog.ErrorS(errors.New("request end"), string(respBody.ResponseBody.GetBody()), "requestID", routerCtx.RequestID)
				klog.Errorf("Response body processing error %d, requestID: %s, selectedPod: %s, model: %s", respErrorCode, routerCtx.RequestID, routerCtx.TargetAddress(), model)
				resp = generateErrorResponse(envoyTypePb.StatusCode(respErrorCode), nil, string(respBody.ResponseBody.GetBody()))
				// Clean up inflight counters for the failed request.
				// The backend rejected the request (e.g., 400 context length exceeded),
				// so HandleResponseBody is never called and counters would leak.
				// Guard with HasRouted(): if routing failed in HandleRequestBody, no
				// inflight counters were incremented and cleanup would underflow.
				if !inflightCleaned && routerCtx != nil && routerCtx.HasRouted() {
					klog.V(5).InfoS("Cleaning up inflight counters for error response",
						"requestID", routerCtx.RequestID,
						"errorCode", respErrorCode,
						"model", model)
					s.cleanupInflightCounters(routerCtx, routerCtx.RequestID)
					s.cleanupPerRequestState(routerCtx, routerCtx.RequestID)
					inflightCleaned = true
				}
			} else {
				prevCompleted := completed
				resp, completed = s.HandleResponseBody(ctx, req, user, rpm, model, stream, traceTerm, completed)
				if !prevCompleted && completed {
					// DoneRequestTrace was called inside HandleResponseBody's defer.
					requestCounted = false
					// Only mark inflight as cleaned if the EndOfStream path actually ran
					// (which deletes RequestTimings after decrementing counters via
					// calculateTimingMetrics). Error paths in HandleResponseBody may
					// set complete=true without cleaning inflight counters; in that case
					// the top-level defer must still run cleanupInflightCounters.
					if _, exists := utils.RequestTimings.Load(routerCtx.RequestID); !exists {
						inflightCleaned = true
					}
				}
			}
		default:
			klog.Errorf("Unknown Request type %+v\n", v)
		}

		if err := srv.Send(resp); err != nil {
			// context.Canceled typically means Envoy already finished delivering
			// the response to the client and tore down the ext_proc stream.
			// This is benign in streamed response mode — the client got all data.
			if ctx.Err() == context.Canceled {
				klog.V(4).InfoS("srv.Send failed after context canceled (client/Envoy finished), ending stream", "requestID", requestID)
			} else {
				klog.ErrorS(err, "srv.Send failed, ending stream", "requestID", requestID)
			}
			// Return immediately — the top-level defer handles cleanup
			// (DoneRequestCount if requestCounted is still true, and routerCtx.Delete).
			return fmt.Errorf("srv.Send failed for request %s: %w", requestID, err)
		}
	}
}

func (s *Server) selectTargetPod(ctx *types.RoutingContext, pods types.PodList) (string, error) {
	// klog.Infof("selectTargetPod starts. context state, requestID: %s, ctx.Err(): %v", ctx.RequestID, ctx.Err())
	defer func() {
		if ctx.Err() != nil {
			klog.ErrorS(ctx.Err(), "Exiting selectTargetPod, Context error", "requestID", ctx.RequestID)
		} else {
			klog.V(5).InfoS("Exiting selectTargetPod, Context is not done successfully", "requestID", ctx.RequestID)
		}
	}()

	router, err := routing.Select(ctx.Algorithm)(ctx)
	if err != nil {
		klog.ErrorS(err, "Router selection failed", "requestID", ctx.RequestID)
		return "", err
	}

	if pods.Len() == 0 {
		return "", fmt.Errorf("no pods to forward request")
	}
	readyPods := utils.FilterRoutablePods(pods.All())
	if len(readyPods) == 0 {
		return "", fmt.Errorf("no ready pods available for fallback")
	}
	if len(readyPods) == 1 {
		for _, pod := range readyPods {
			ctx.SetTargetPod(pod)
			return ctx.TargetAddress(), nil
		}
	}

	for _, pod := range readyPods {
		utils.MetricsTracker.InitPodKey(pod.Status.PodIP)
	}
	utils.SyncPodRegistry(readyPods)

	klog.V(5).Infof("selectTargetPod, done with InitPodKey. context state, requestID: %s, ctx.Err(): %v", ctx.RequestID, ctx.Err())

	ts := time.Now()
	selectedPodAddress, err := router.Route(ctx, &utils.PodArray{Pods: readyPods})

	klog.V(5).Infof("selectTargetPod, requestID: %s, Routing took %s, selectedPodAddress: %s", ctx.RequestID, time.Since(ts), selectedPodAddress)
	if err != nil {
		klog.ErrorS(err, "Routing failed", "requestID", ctx.RequestID)
		return "", err
	}
	return selectedPodAddress, nil
}

// cleanupInflightCounters decrements inflight counters for a request that didn't
// complete normally through calculateTimingMetrics. It checks RequestTiming to
// determine whether the request was still in the prefill phase or had transitioned
// to decode, and decrements accordingly.
//
// Idempotency: This method uses LoadAndDelete on RequestTimings, so it can safely
// be called multiple times — only the first call does actual work. This is critical
// because multiple exit paths (streaming error handlers, HandleResponseBody error
// returns, top-level defer) may all attempt cleanup.
func (s *Server) cleanupInflightCounters(routerCtx *types.RoutingContext, requestID string) {
	if routerCtx == nil {
		return
	}
	podIP := routerCtx.TargetAddressWithoutPort()
	if podIP == "" {
		return
	}

	timingObj, loaded := utils.RequestTimings.LoadAndDelete(requestID)
	if !loaded {
		return
	}
	timing, ok := timingObj.(*RequestTiming)
	if !ok || timing == nil {
		return
	}

	if timing.firstTokenTime.IsZero() {
		// Still in prefill phase — backend rejected before generating any tokens
		utils.DecrementNumInflightPrefillRequestsForPod(podIP)
		utils.DecrementNumInflightPrefillTokensForPod(podIP, int(timing.prefillTokenCount))
	} else {
		// Had transitioned to decode phase
		utils.DecrementNumInflightDecodeRequestsForPod(podIP)
		utils.DecrementNumInflightDecodeTokensForPod(podIP, int(timing.decodeTokenCount))
	}
	utils.DecrementNumInflightRequestsForPod(podIP)

	klog.V(5).Infof("cleanupInflightCounters: decremented counters, requestID: %s, podIP: %s, wasPrefill: %t, prefillTokens: %d, decodeTokens: %d",
		"requestID", requestID,
		"podIP", podIP,
		"wasPrefill", timing.firstTokenTime.IsZero(),
		"prefillTokens", timing.prefillTokenCount,
		"decodeTokens", timing.decodeTokenCount)
}

// cleanupPerRequestState removes all per-request tracking state from sync.Maps.
// Safe to call even if some entries were never created (Delete on missing key is a no-op).
func (s *Server) cleanupPerRequestState(routerCtx *types.RoutingContext, requestID string) {
	if requestID == "" {
		return
	}

	// RL agent cleanup
	if routerCtx != nil && routerCtx.SubAlgorithm == "scalable_rl_agent" {
		utils.RemoveLiveRequest(requestID)
	}

	// Per-request metric snapshots
	utils.CleanupKVCacheHitRatio(requestID)
	utils.CleanupInflightRequests(requestID)
	utils.CleanupvLLMGPUKVCacheUsage(requestID)
	utils.CleanupvLLMNumRequestsRunning(requestID)
	utils.CleanupvLLMNumRequestsWaiting(requestID)
	utils.CleanupSnapshotNumInflightPrefillTokensForRequest(requestID)
	utils.CleanupSnapshotNumInflightDecodeTokensForRequest(requestID)
	utils.CleanupSnapshotNumInflightPrefillRequestsForRequest(requestID)
	utils.CleanupSnapshotNumInflightDecodeRequestsForRequest(requestID)
	utils.CleanupRequestPodMetrics(requestID)

	// Token and routing data
	utils.CleanupByteArrayPrefillTokensForRequest(requestID)
	utils.CleanupHashOfPrefixHashesForRequest(requestID)
	utils.CleanupNumPrefillTokensForRequest(requestID)
	utils.CleanupNumDecodeTokensForRequest(requestID)
	utils.CleanuprequestToPodIP(requestID)

	// Routing algorithm state
	utils.CleanupExploration(requestID)
	utils.CleanupPredictedLatencies(requestID)
	utils.CleanupChosenPodPredictedLatency(requestID)
	utils.CleanupPredictedRewards(requestID)
	utils.CleanupChosenPodPredictedReward(requestID)
	utils.CleanupSelectedPodGPU(requestID)
	utils.CleanupOODFallbackForRequest(requestID)
	utils.CleanupFailureFallbackForRequest(requestID)
	utils.CleanupPrevRewardForRequest(requestID)

	// Server-local per-request maps
	// Note: RequestTimings is intentionally NOT deleted here. It is consumed by
	// cleanupInflightCounters via LoadAndDelete to ensure exactly-once decrement.
	s.routingContexts.Delete(requestID)
	s.requestHeaders.Delete(requestID)
	s.streamingUsageCache.Delete(requestID)
	s.requestBuffers.Delete(requestID)
	s.statusCode.Delete(requestID)
	s.selectedPodIP.Delete(requestID)
}

func NewHealthCheckServer() *HealthServer {
	return &HealthServer{}
}

type HealthServer struct{}

func (s *HealthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *HealthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "watch is not implemented")
}
