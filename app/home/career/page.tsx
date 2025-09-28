'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Briefcase, ChevronDown } from 'lucide-react'

// --- Types ---
type Course = {
  id: string
  name: string
  credits: number
  dept: string
}

type Career = {
  title: string
  probability: number
  confidence: 'High' | 'Medium' | 'Low'
  reasoning: string
  recommendedCourses: Course[]
}

export default function CareerPage() {
  const [careers, setCareers] = useState<Career[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    // Mock FastAPI fetch — replace with real API
    const mockCareers: Career[] = [
      {
        title: "Data Scientist",
        probability: 0.82,
        confidence: "High",
        reasoning:
          "Strong foundation in AI and distributed systems; aligns well with data-driven problem solving.",
        recommendedCourses: [
          { id: "CSDS 400", name: "Advanced Machine Learning", credits: 4, dept: "CS" },
          { id: "CSDA 300", name: "Data Mining", credits: 3, dept: "CS" },
          { id: "STAT 350", name: "Statistical Inference", credits: 3, dept: "Statistics" },
        ],
      },
      {
        title: "Bioinformatics Researcher",
        probability: 0.67,
        confidence: "Medium",
        reasoning:
          "Combination of Biology and CS suggests potential in computational biology.",
        recommendedCourses: [
          { id: "BIOI 320", name: "Genomics Data Analysis", credits: 3, dept: "Biology" },
          { id: "CSBB 310", name: "Algorithms for Bioinformatics", credits: 3, dept: "CS" },
        ],
      },
      {
        title: "Software Engineer",
        probability: 0.9,
        confidence: "High",
        reasoning:
          "Core CS courses and AI indicate readiness for software development roles.",
        recommendedCourses: [
          { id: "CSSE 210", name: "Software Engineering Principles", credits: 3, dept: "CS" },
          { id: "CSWC 300", name: "Web & Cloud Development", credits: 3, dept: "CS" },
        ],
      },
    ]
    setCareers(mockCareers)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 p-8 md:p-12">
      {/* Header */}
      <motion.h1
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-3xl md:text-4xl font-extrabold text-gray-900 mb-3"
      >
        Career Path Predictions
      </motion.h1>
      <p className="text-gray-500 mb-10">
        Based on your completed courses, here are possible career paths with
        probabilities, confidence levels, reasoning, and recommended next steps.
      </p>

      {/* Career List */}
      <div className="space-y-6">
        {careers.map((career) => {
          const isOpen = expanded === career.title
          return (
            <motion.div
              key={career.title}
              className="bg-white rounded-xl shadow-md border border-gray-200 p-6 transition hover:shadow-lg"
            >
              {/* Header */}
              <div
                className="flex justify-between items-center cursor-pointer"
                onClick={() => setExpanded(isOpen ? null : career.title)}
              >
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                    <Briefcase className="w-5 h-5 text-blue-600 mr-2" />
                    {career.title}
                  </h2>
                  <p className="text-sm text-gray-500">
                    Probability:{" "}
                    <span className="font-medium text-blue-600">
                      {(career.probability * 100).toFixed(1)}%
                    </span>{" "}
                    | Confidence:{" "}
                    <span
                      className={`font-medium ${
                        career.confidence === "High"
                          ? "text-green-600"
                          : career.confidence === "Medium"
                          ? "text-amber-600"
                          : "text-red-600"
                      }`}
                    >
                      {career.confidence}
                    </span>
                  </p>
                </div>
                <ChevronDown
                  className={`w-5 h-5 text-gray-500 transition-transform ${
                    isOpen ? "rotate-180" : ""
                  }`}
                />
              </div>

              {/* Expandable Details */}
              {isOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mt-4 text-gray-600 text-sm"
                >
                  {/* Reasoning */}
                  <p className="mb-4">{career.reasoning}</p>

                  {/* Recommended Courses */}
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Recommended Courses
                  </h3>
                  <div className="flex space-x-4 overflow-x-auto hide-scrollbar pb-2">
                    {career.recommendedCourses.map((c) => (
                      <div
                        key={c.id}
                        className="min-w-[220px] bg-gray-50 border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition"
                      >
                        <p className="font-mono text-xs text-gray-500">
                          {c.id}
                        </p>
                        <h4 className="font-semibold text-gray-800">
                          {c.name}
                        </h4>
                        <p className="text-xs text-gray-500 mt-1">
                          {c.dept} • {c.credits} credits
                        </p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
