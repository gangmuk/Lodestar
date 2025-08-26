

gateway_req_body.go -> selectTargetPod

rl_routing.go -> Route() 
Route() 
  - prefix = MatchPrefix(input prompt)
  - SetHashOfPrefixHashesForRequest(prefix)
    - hash = hash_prefixhashes(prefix)
    - requestToHashOfPrefixHashes[requestID] = hash
  - numOutputTokens, _ := utils.GetNumOutputTokensForPrefixHashes(prefix)
    - hash = hash_prefixhashes(prefix)
    - numOutputTokens = hashToNumOutputTokens[hash]


gateway_req_body.go -> vllm pod

vllm pod -> gateway_rsp_body.go

gateway_rsp_body.go -> utils.SetNumOutputTokensForRequest(routerCtx.RequestID, int(usage.CompletionTokens))
- SetNumOutputTokensForRequest(requestID, numOutputTokens)
  - hash  = requestToHashOfPrefixHashes[requestID]
  - hashToNumOutputTokens[hash] = numOutputTokens