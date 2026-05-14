package com.soundpulse.runtime

import java.util.ArrayDeque

class AudioWindowSegmenter(
    private val windowSamples: Int,
    private val hopSamples: Int,
) {
    private val queue = ArrayDeque<Float>()

    init {
        require(windowSamples > 0) { "windowSamples must be > 0" }
        require(hopSamples > 0) { "hopSamples must be > 0" }
    }

    fun reset() {
        queue.clear()
    }

    fun append(samples: FloatArray): List<FloatArray> {
        for (sample in samples) {
            queue.addLast(sample)
        }

        val windows = mutableListOf<FloatArray>()
        while (queue.size >= windowSamples) {
            val window = FloatArray(windowSamples)
            var index = 0
            for (value in queue) {
                if (index == windowSamples) {
                    break
                }
                window[index] = value
                index += 1
            }

            windows.add(window)
            val dropped = hopSamples.coerceAtMost(queue.size)
            repeat(dropped) {
                queue.removeFirst()
            }
        }

        return windows
    }
}
