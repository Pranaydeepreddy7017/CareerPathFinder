'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BookOpen,
  CheckCircle,
  Clock,
  Zap,
  ChevronDown,
} from 'lucide-react'

// --- Data Types and Mock Data ---

type Course = {
  id: string
  name: string
  credits: number
  dept: string
  difficulty?: string
  label?: string
  prereqs?: string
}

const takenCourses: Course[] = [
  { id: "BEEE 200", name: "Cell Physiology", credits: 3, dept: "Biology" },
  { id: "BFFF 100", name: "Introduction to Molecular Genetics", credits: 3, dept: "Biology" },
  { id: "BGGG 100", name: "Introduction to Immunology", credits: 3, dept: "Biology" },
  { id: "CSKK 200", name: "Computer Vision", credits: 1, dept: "Computer Science" },
  { id: "CSOO 200", name: "Distributed Systems", credits: 3, dept: "Computer Science" },
  { id: "CSSS 100", name: "Basic Game Development", credits: 3, dept: "Computer Science" },
  { id: "CSVV 200", name: "Artificial Intelligence", credits: 3, dept: "Computer Science" },
]

const recommendedCourses: Course[] = [
  {
    id: "CSUU 400",
    name: "Special Topics in Computer Graphics",
    credits: 4,
    dept: "Computer Science",
    difficulty: "Hard",
    label: "Medium Success",
    prereqs: "CSJJ 300 (Compiler Design Methods)",
  },
  {
    id: "CSZZ 100",
    name: "Principles of DevOps",
    credits: 2,
    dept: "Computer Science",
    difficulty: "Medium",
    label: "High Success",
    prereqs: "None",
  },
  {
    id: "BWWW 300",
    name: "Proteomics Theory",
    credits: 3,
    dept: "Biology",
    difficulty: "Hard",
    label: "Medium Success",
    prereqs: "BCCC 100",
  },
]

// --- Helper Functions ---
const getDifficultyColors = (difficulty?: string) => {
  switch (difficulty) {
    case 'Hard':
      return 'bg-red-100 text-red-700 border-red-300'
    case 'Medium':
      return 'bg-amber-100 text-amber-700 border-amber-300'
    case 'Easy':
      return 'bg-green-100 text-green-700 border-green-300'
    default:
      return 'bg-gray-100 text-gray-600 border-gray-300'
  }
}

// Card animation
const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

// --- Main Component ---
export default function RecommendationPage() {
  const [expandedId, setExpandedId] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)

  const handlePredict = (courseId: string) => {
    if (expandedId === courseId) {
      setExpandedId(null)
      setPrediction(null)
      return
    }
    setExpandedId(courseId)
    setPrediction(null)

    // Mock API delay
    setTimeout(() => {
      setPrediction(`Predicted Grade: A- / Success Probability: 92%`)
    }, 700)
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6 md:p-12">
      {/* --- Header --- */}
      <motion.h1
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-4xl font-extrabold text-gray-900 mb-2"
      >
        Your Academic Pathfinder
      </motion.h1>
      <p className="text-gray-500 mb-10">
        Review your progress and explore personalized course recommendations.
      </p>

      {/* --- Courses Taken Section --- */}
      <section className="mb-14">
        <h2 className="text-2xl font-semibold text-gray-700 mb-6 flex items-center">
          <CheckCircle className="w-6 h-6 text-green-500 mr-2" />
          Completed Courses ({takenCourses.length})
        </h2>
        <motion.div
          initial="hidden"
          animate="visible"
          transition={{ staggerChildren: 0.07 }}
          className="flex space-x-4 md:space-x-6 overflow-x-auto pb-4 hide-scrollbar"
        >
          {takenCourses.map((c) => (
            <motion.div
              key={c.id}
              variants={cardVariants}
              className="min-w-[240px] flex-shrink-0 bg-white rounded-xl shadow-lg p-5 flex flex-col justify-between border-t-4 border-green-500/80 transition-shadow duration-300 hover:shadow-xl"
            >
              <div className="flex justify-between items-start mb-3">
                <p className="font-mono text-sm text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                  {c.id}
                </p>
                <Clock className="w-5 h-5 text-green-500 opacity-80" />
              </div>
              <h3 className="font-bold text-lg text-gray-800 line-clamp-2">
                {c.name}
              </h3>
              <div className="mt-4 pt-3 border-t border-gray-100 flex justify-between text-sm">
                <p className="text-gray-600 font-medium">{c.dept}</p>
                <p className="font-bold text-green-600">{c.credits} credits</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* --- Recommended Courses Section --- */}
      <section>
        <h2 className="text-2xl font-semibold text-gray-700 mb-6 flex items-center">
          <BookOpen className="w-6 h-6 text-blue-500 mr-2" />
          Recommended Courses ({recommendedCourses.length})
        </h2>
        <motion.div
          initial="hidden"
          animate="visible"
          transition={{ staggerChildren: 0.07, delayChildren: 0.2 }}
          className="flex space-x-4 md:space-x-6 overflow-x-auto pb-4 hide-scrollbar"
        >
          {recommendedCourses.map((c) => {
            const isExpanded = expandedId === c.id
            const difficultyClasses = getDifficultyColors(c.difficulty)

            return (
              <motion.div
                key={c.id}
                variants={cardVariants}
                onClick={() => handlePredict(c.id)}
                className={`
                  min-w-[280px] flex-shrink-0 bg-white rounded-xl shadow-xl p-6 flex flex-col border-t-4
                  ${isExpanded ? 'border-blue-600 shadow-2xl scale-[1.01]' : 'border-blue-400 hover:shadow-2xl hover:border-blue-500'}
                  transition-all duration-300 cursor-pointer
                `}
              >
                {/* Header and ID */}
                <div className="flex justify-between items-center mb-3">
                  <p className="font-mono text-xs text-gray-500 bg-blue-50 px-2 py-0.5 rounded font-medium">
                    {c.id}
                  </p>
                  <Zap className="w-5 h-5 text-blue-500" />
                </div>

                {/* Course Name */}
                <h3 className="font-bold text-xl text-gray-900 line-clamp-2">
                  {c.name}
                </h3>
                <p className="text-sm text-gray-500 mt-1 mb-3">{c.dept}</p>

                {/* Metadata Tags */}
                <div className="flex space-x-2 my-2">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${difficultyClasses} border`}>
                    {c.difficulty}
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-700">
                    {c.credits} Credits
                  </span>
                </div>

                {/* Prereqs & CTA */}
                <p className="text-sm text-gray-500 mt-3 italic border-t pt-3 border-gray-100">
                  <span className="font-medium text-gray-700">Prereqs:</span> {c.prereqs || 'None'}
                </p>
                
                <button
                  className="mt-4 flex items-center justify-between text-blue-600 font-semibold pt-4 border-t border-dashed border-blue-100"
                  aria-expanded={isExpanded}
                  aria-controls={`prediction-${c.id}`}
                >
                  Click for Personalized Prediction
                  <ChevronDown
                    className={`w-5 h-5 transition-transform duration-300 ${isExpanded ? 'rotate-180' : 'rotate-0'}`}
                  />
                </button>

                {/* Expandable prediction */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      id={`prediction-${c.id}`}
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.4 }}
                      className="mt-3 p-3 bg-blue-50 rounded-lg text-sm border border-blue-200"
                    >
                      {prediction ? (
                        <p className="text-blue-800 font-bold flex items-center">
                          <Zap className="w-4 h-4 mr-2 text-blue-600" />
                          {prediction}
                        </p>
                      ) : (
                        <p className="text-blue-500 italic">
                          Calculating your success prediction...
                        </p>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </motion.div>
      </section>
    </div>
  )
}
