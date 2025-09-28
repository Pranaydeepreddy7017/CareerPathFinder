"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { GraduationCap, User, Briefcase } from "lucide-react";

export default function WhoPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex flex-col items-center justify-center px-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-12"
      >
        <div className="flex justify-center mb-6">
          <GraduationCap className="w-14 h-14 text-blue-600" />
        </div>
        <h1 className="text-4xl font-extrabold text-gray-900 mb-4">
          Who’s using <span className="text-blue-600">Student Career Pathfinder</span>?
        </h1>
        <p className="text-gray-600 text-lg">
          Select your role to get started with personalized insights.
        </p>
      </motion.div>

      {/* Role Selection */}
      <div className="flex gap-12">
        <Link href="/student">
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-white shadow-lg rounded-2xl p-10 w-64 text-center border border-gray-100 hover:shadow-xl transition cursor-pointer"
          >
            <User className="w-12 h-12 text-green-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Student</h2>
            <p className="text-gray-500 text-sm">
              Access your profile, recommendations, and progress dashboard.
            </p>
          </motion.div>
        </Link>

        <Link href="/home/advisor">
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="bg-white shadow-lg rounded-2xl p-10 w-64 text-center border border-gray-100 hover:shadow-xl transition cursor-pointer"
          >
            <Briefcase className="w-12 h-12 text-purple-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Advisor</h2>
            <p className="text-gray-500 text-sm">
              Guide students, view analytics, and track academic progress.
            </p>
          </motion.div>
        </Link>
      </div>

      {/* Footer */}
      <footer className="mt-20 text-sm text-gray-400">
        © {new Date().getFullYear()} Student Career Pathfinder · Designed for excellence 
      </footer>
    </div>
  );
}
