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
    fun run(input: Any, output: Any)
    fun getInputShape(): IntArray
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

class TinyCnnEmbeddingExtractor(
    private val interpreter: LiteRtInterpreter,
    preprocessing: PreprocessingConfig,
    private val embeddingDim: Int,
) : EmbeddingExtractor {
    private val featureBuilder: TinyCnnFeatureBuilder

    init {
        val shape = interpreter.getInputShape()
        val height = if (shape.size >= 3) shape[1] else 96
        val width = if (shape.size >= 3) shape[2] else 101
        val channels = if (shape.size >= 4) shape[3] else 1
        featureBuilder = TinyCnnFeatureBuilder(
            config = preprocessing,
            imageHeight = height,
            imageWidth = width,
            imageChannels = channels
        )
    }

    override fun extractEmbedding(preprocessedWindow: FloatArray): FloatArray {
        val output = FloatArray(embeddingDim)
        interpreter.run(featureBuilder.buildImageTensor(preprocessedWindow), output)
        return output
    }
}

private class ReflectionLiteRtInterpreter(
    modelBytes: ByteArray,
) : LiteRtInterpreter {
    private val interpreterClass: Class<*>
    private val interpreterInstance: Any
    private val runMethod: java.lang.reflect.Method
    private val getInputTensorMethod: java.lang.reflect.Method
    private val getOutputTensorMethod: java.lang.reflect.Method
    private val resizeInputMethod: java.lang.reflect.Method
    private val allocateTensorsMethod: java.lang.reflect.Method
    private val tensorClass: Class<*>
    private val shapeMethod: java.lang.reflect.Method

    private var lastInputSize: Int = -1

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

        getInputTensorMethod = interpreterClass.getMethod("getInputTensor", Int::class.javaPrimitiveType)
        getOutputTensorMethod = interpreterClass.getMethod("getOutputTensor", Int::class.javaPrimitiveType)
        resizeInputMethod = interpreterClass.getMethod("resizeInput", Int::class.javaPrimitiveType, IntArray::class.java)
        allocateTensorsMethod = interpreterClass.getMethod("allocateTensors")
        tensorClass = Class.forName("org.tensorflow.lite.Tensor")
        shapeMethod = tensorClass.getMethod("shape")
    }

    override fun getInputShape(): IntArray {
        val inputTensor = getInputTensorMethod.invoke(interpreterInstance, 0)
        return shapeMethod.invoke(inputTensor) as IntArray
    }

    override fun run(input: Any, output: Any) {
        val inputTensor = getInputTensorMethod.invoke(interpreterInstance, 0)
        val inputShape = shapeMethod.invoke(inputTensor) as IntArray

        if (input is FloatArray && inputShape.size == 1) {
            val currentShapeVal = inputShape[0]
            if (input.size != currentShapeVal || input.size != lastInputSize) {
                resizeInputMethod.invoke(interpreterInstance, 0, intArrayOf(input.size))
                allocateTensorsMethod.invoke(interpreterInstance)
                lastInputSize = input.size
            }
        }

        val outputTensor = getOutputTensorMethod.invoke(interpreterInstance, 0)
        val outputShape = shapeMethod.invoke(outputTensor) as IntArray

        if (outputShape.size == 2) {
            val numFrames = outputShape[0]
            val embeddingDim = outputShape[1]
            val tempOutput = Array(numFrames) { FloatArray(embeddingDim) }

            runMethod.invoke(interpreterInstance, input, tempOutput)

            val outputArray = output as FloatArray
            for (j in 0 until minOf(embeddingDim, outputArray.size)) {
                var sum = 0f
                for (i in 0 until numFrames) {
                    sum += tempOutput[i][j]
                }
                outputArray[j] = sum / numFrames
            }
        } else {
            runMethod.invoke(interpreterInstance, input, output)
        }
    }
}
