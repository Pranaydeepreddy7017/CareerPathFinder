"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { GraduationCap } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex flex-col items-center justify-center text-center px-6">
      <motion.div
        initial={{ opacity: 0, y: -40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-3xl"
      >
        <GraduationCap className="w-20 h-20 text-blue-600 mx-auto mb-6" />
        <h1 className="text-6xl font-extrabold text-gray-900 mb-4">
          Student Career <span className="text-blue-600">Pathfinder</span>
        </h1>
        <p className="text-lg text-gray-600 mb-10">
          Your personalized academic journey. Smarter recommendations. Better insights.  
        </p>
        <Link
          href="/who"
          className="px-8 py-3 bg-blue-600 text-white text-lg font-semibold rounded-lg hover:bg-blue-700 transition"
        >
          Get Started →
        </Link>
      </motion.div>
    </div>
  );
}
