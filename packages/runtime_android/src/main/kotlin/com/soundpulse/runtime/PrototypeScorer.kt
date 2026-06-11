package com.soundpulse.runtime

import kotlin.math.max

class PrototypeScorer(
    prototypes: Map<String, FloatArray>,
) {
    private val originalPrototypes: Map<String, FloatArray> = prototypes.mapValues { it.value.copyOf() }
    private val prototypes: MutableMap<String, FloatArray> = prototypes.mapValues { it.value.copyOf() }.toMutableMap()

    init {
        require(prototypes.isNotEmpty()) { "prototypes cannot be empty" }
    }

    fun updatePrototype(label: String, embedding: FloatArray) {
        require(label.isNotBlank()) { "label cannot be blank" }
        require(embedding.isNotEmpty()) { "prototype embedding cannot be empty" }
        prototypes[label] = embedding.copyOf()
    }

    fun resetPrototype(label: String) {
        val original = originalPrototypes[label]
        if (original != null) {
            prototypes[label] = original.copyOf()
        } else {
            prototypes.remove(label)
        }
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
        return sum
    }
}
