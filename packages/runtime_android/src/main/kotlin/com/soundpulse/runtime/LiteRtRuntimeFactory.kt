package com.soundpulse.runtime

import java.io.File

object LiteRtRuntimeFactory {
    fun createFromArtifacts(
        artifacts: RuntimeArtifacts,
        embeddingExtractor: EmbeddingExtractor,
        policy: RuntimePolicy = RuntimePolicy(),
    ): LiteRtRuntimeEngine {
        return LiteRtRuntimeEngine(
            preprocessing = artifacts.preprocessing,
            thresholds = artifacts.thresholds,
            embeddingExtractor = embeddingExtractor,
            prototypeScorer = PrototypeScorer(artifacts.prototypes),
            policy = policy,
        )
    }

    fun createWithLiteRtBackbones(
        artifacts: RuntimeArtifacts,
        tinyCnnModelFile: File,
        yamnetModelFile: File,
        activeBackbone: RuntimeBackbone = RuntimeBackbone.TINY_CNN,
        policy: RuntimePolicy = RuntimePolicy(),
    ): Pair<LiteRtRuntimeEngine, MultiBackboneEmbeddingExtractor> {
        val tinyInterpreter = LiteRtModelLoader.loadInterpreter(tinyCnnModelFile)
        val yamnetInterpreter = LiteRtModelLoader.loadInterpreter(yamnetModelFile)
        val extractor =
            MultiBackboneEmbeddingExtractor(
                tinyCnnExtractor =
                    TinyCnnEmbeddingExtractor(
                        interpreter = tinyInterpreter,
                        preprocessing = artifacts.preprocessing,
                        embeddingDim = artifacts.preprocessing.tinyCnnEmbeddingDim,
                    ),
                yamnetExtractor =
                    LiteRtEmbeddingExtractor(
                        interpreter = yamnetInterpreter,
                        embeddingDim = artifacts.preprocessing.yamnetEmbeddingDim,
                    ),
                activeBackbone = activeBackbone,
            )

        val engine = createFromArtifacts(artifacts, extractor, policy)
        return engine to extractor
    }
}
