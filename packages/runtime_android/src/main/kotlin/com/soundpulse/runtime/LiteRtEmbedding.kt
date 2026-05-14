package com.soundpulse.runtime

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class RuntimeBackbone {
    TINY_CNN,
    YAMNET,
}

interface EmbeddingExtractor {
    fun extractEmbedding(preprocessedWindow: FloatArray): FloatArray
}

class MultiBackboneEmbeddingExtractor(
    private val tinyCnnExtractor: EmbeddingExtractor,
    private val yamnetExtractor: EmbeddingExtractor,
    private var activeBackbone: RuntimeBackbone = RuntimeBackbone.TINY_CNN,
) : EmbeddingExtractor {
    fun setActiveBackbone(backbone: RuntimeBackbone) {
        activeBackbone = backbone
    }

    override fun extractEmbedding(preprocessedWindow: FloatArray): FloatArray {
        return when (activeBackbone) {
            RuntimeBackbone.TINY_CNN -> tinyCnnExtractor.extractEmbedding(preprocessedWindow)
            RuntimeBackbone.YAMNET -> yamnetExtractor.extractEmbedding(preprocessedWindow)
        }
    }
}

interface LiteRtInterpreter {
    fun run(input: FloatArray, output: FloatArray)
}

object LiteRtModelLoader {
    fun loadInterpreter(modelFile: File): LiteRtInterpreter {
        require(modelFile.exists()) { "LiteRT model file not found: ${modelFile.path}" }
        val modelBytes = modelFile.readBytes()
        return ReflectionLiteRtInterpreter(modelBytes)
    }
}

class LiteRtEmbeddingExtractor(
    private val interpreter: LiteRtInterpreter,
    private val embeddingDim: Int,
) : EmbeddingExtractor {
    override fun extractEmbedding(preprocessedWindow: FloatArray): FloatArray {
        val output = FloatArray(embeddingDim)
        interpreter.run(preprocessedWindow, output)
        return output
    }
}

private class ReflectionLiteRtInterpreter(
    modelBytes: ByteArray,
) : LiteRtInterpreter {
    private val interpreterClass: Class<*>
    private val interpreterInstance: Any
    private val runMethod: java.lang.reflect.Method

    init {
        interpreterClass = Class.forName("org.tensorflow.lite.Interpreter")
        val buffer =
            ByteBuffer
                .allocateDirect(modelBytes.size)
                .order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)
        buffer.rewind()

        val constructor = interpreterClass.getConstructor(ByteBuffer::class.java)
        interpreterInstance = constructor.newInstance(buffer)
        runMethod = interpreterClass.getMethod("run", Any::class.java, Any::class.java)
    }

    override fun run(input: FloatArray, output: FloatArray) {
        runMethod.invoke(interpreterInstance, input, output)
    }
}
