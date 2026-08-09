# MTP research protocol

## Do not rank MTP by its raw auxiliary loss

At random initialization a 32,768-way token predictor has a uniform cross-entropy of `ln(32768) ≈ 10.397`. Predicting t+2/t+3 is harder than next-token prediction, so a large early MTP loss is expected. The useful question is whether the auxiliary task improves the **main model** after training enough for the MTP module to learn.

The old synthetic profiler used random token IDs and random labels. Its MTP loss can measure numerical stability, memory and runtime overhead, but **cannot measure MTP usefulness**.

## vNext2 comparisons

On the selected proxy architecture and LR, start every variant from scratch on the same real contiguous text and fixed token budget:

- no MTP;
- Aster low-rank MTP depth 1;
- DeepSeek-style sequential MTP depth 1.

Rank by:

1. main validation loss (`eval_main_loss`) at fixed model tokens;
2. wall-clock/tokens-per-second cost;
3. MTP auxiliary trajectory as a diagnostic, not the objective;
4. later, speculative acceptance/latency after a production incremental drafter exists.

Never compare total loss across MTP0 and MTP>0; total loss includes the auxiliary term by construction.

## DeepSeek-style alignment

For source hidden state `h_i`, the one-step MTP module receives the **actual embedding of token `x_{i+1}`**, combines normalized embedding and normalized hidden through a 2d→d projection, runs a full Aster block, final-normalizes, and predicts `x_{i+2}` through the shared output head. Document-boundary masks are propagated so MTP cannot learn an arbitrary transition across packed documents.

Aster vNext2 intentionally supports one sequential MTP layer only. The production speculative path is a separate problem: during inference the future token is a proposal, not a known teacher token, and the MTP layer needs incremental state/cache management. vNext2 refuses to present full-prefix re-verification as a speed optimization.
