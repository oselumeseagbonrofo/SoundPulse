package com.soundpulse.runtime

import kotlin.math.max

class PrototypeScorer(
    private val prototypes: Map<String, FloatArray>,
) {
    init {
        require(prototypes.isNotEmpty()) { "prototypes cannot be empty" }
    }

    fun score(embedding: FloatArray, topN: Int): List<Candidate> {
        require(topN > 0) { "topN must be > 0" }

        val scored = mutableListOf<Candidate>()
        for ((label, prototype) in prototypes) {
            val distance = squaredL2(embedding, prototype)
            scored.add(
                Candidate(
                    label = label,
                    distance = distance,
                    confidence = 0f,
                ),
            )
        }

        val sorted = scored.sortedBy { it.distance }
        if (sorted.isEmpty()) {
            return emptyList()
        }

        val maxDistance = max(sorted.last().distance, 1e-6f)
        return sorted
            .take(topN)
            .map { candidate ->
                val confidence = (1f - (candidate.distance / maxDistance)).coerceIn(0f, 1f)
                candidate.copy(confidence = confidence)
            }
    }

    private fun squaredL2(left: FloatArray, right: FloatArray): Float {
        val length = left.size.coerceAtMost(right.size)
        if (length == 0) {
            return Float.MAX_VALUE
        }

        var sum = 0f
        for (index in 0 until length) {
            val delta = left[index] - right[index]
            sum += delta * delta
        }
        return sum / length
    }
}
