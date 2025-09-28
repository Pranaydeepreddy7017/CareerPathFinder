"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ResponsivePie } from "@nivo/pie";
import { ResponsiveAreaBump } from "@nivo/bump";

// Types
type StudentData = {
  studentId: string;
  degree: string;
  totalCredits: number;
  creditsCompleted: number;
  creditsRemaining: number;
  recommendedCourses?: any[];
};

export default function EnhancedProfile() {
  const [student, setStudent] = useState<StudentData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("studentData");
    if (stored) {
      const parsed = JSON.parse(stored);
      console.log("Parsed studentData:", parsed);

      if (parsed.student_id) {
        fetch(
          `http://127.0.0.1:8000/recommend?student_id=${parsed.student_id}&term=Spring2024`
        )
          .then((res) => res.json())
          .then((data) => {
            console.log("API response:", data);
            setStudent(data);
          })
          .catch((err) => console.error("Error fetching:", err))
          .finally(() => setLoading(false));
      } else {
        console.error("schoolId missing in stored:", parsed);
        setLoading(false);
      }
    } else {
      console.error("studentData not found in localStorage");
      setLoading(false);
    }
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-500">Loading profile...</p>
      </div>
    );
  }

  if (!student) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-red-500">No student data found</p>
      </div>
    );
  }

  // Hero stats
  const heroStats = [
    {
      label: "Total Credits",
      value: student.totalCredits,
      color: "#2563eb",
    },
    {
      label: "Completed",
      value: student.creditsCompleted,
      color: "#16a34a",
    },
    {
      label: "Remaining",
      value: student.creditsRemaining,
      color: "#dc2626",
    },
  ];

  const pieData = [
    {
      id: "Completed",
      label: "Completed",
      value: student.creditsCompleted,
      color: "#16a34a",
    },
    {
      id: "Remaining",
      label: "Remaining",
      value: student.creditsRemaining,
      color: "#d1d5db",
    },
  ];

  // mock comparison (can replace with backend later)
  const comparisonData = [
    {
      id: "You",
      data: [
        { x: "Sem 1", y: 12 },
        { x: "Sem 2", y: student.creditsCompleted },
        { x: "Sem 3", y: student.creditsCompleted + 5 },
        { x: "Sem 4", y: student.creditsCompleted + 10 },
      ],
    },
    {
      id: "Program Average",
      data: [
        { x: "Sem 1", y: 15 },
        { x: "Sem 2", y: 25 },
        { x: "Sem 3", y: 38 },
        { x: "Sem 4", y: 50 },
      ],
    },
  ];

  const percent =
    student.totalCredits > 0
      ? Math.round((student.creditsCompleted / student.totalCredits) * 100)
      : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 text-gray-800 font-sans p-8 pb-20">
      {/* Header */}
      <header className="flex flex-col sm:flex-row sm:justify-between items-center px-4 py-6 bg-white rounded-lg shadow-sm mb-10">
        {/* ID pill */}
        <motion.span
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="font-mono text-sm font-medium px-3 py-1 rounded bg-gray-100 text-gray-700 mr-0 sm:mr-4 mb-3 sm:mb-0"
        >
          ID: {student.studentId}
        </motion.span>

        {/* Program heading */}
        <motion.h1
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-lg md:text-xl font-sans font-normal text-gray-600 tracking-normal text-center sm:text-left"
        >
          {student.degree}
        </motion.h1>

        {/* Student name placeholder */}
        <motion.span
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="font-medium text-teal-700 text-base ml-0 sm:ml-4 mt-3 sm:mt-0"
        >
          Student Profile
        </motion.span>
      </header>

      {/* Stats */}
      <section className="grid md:grid-cols-3 gap-8 mb-12">
        {heroStats.map((stat, i) => (
          <motion.div
            key={i}
            whileHover={{ y: -5, scale: 1.03 }}
            transition={{ type: "spring", stiffness: 200 }}
            className="bg-white rounded-xl shadow-md hover:shadow-xl p-8 flex flex-col items-center border border-gray-200 transition"
          >
            <span className="text-sm font-medium text-gray-500 mb-2 uppercase tracking-wide">
              {stat.label}
            </span>
            <span
              className="text-3xl font-semibold"
              style={{ color: stat.color }}
            >
              {stat.value}
            </span>
          </motion.div>
        ))}
      </section>

      {/* Charts */}
      <section className="grid md:grid-cols-2 gap-10">
        {/* Radial Progress Pie */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="bg-white rounded-xl shadow-md p-8 flex flex-col items-center border border-gray-200"
        >
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Completion Progress
          </h3>
          <div className="relative w-full h-[220px]">
            <ResponsivePie
              data={pieData}
              innerRadius={0.82}
              padAngle={2}
              cornerRadius={6}
              colors={pieData.map((d) => d.color)}
              enableArcLabels={false}
              enableArcLinkLabels={false}
              startAngle={-90}
              endAngle={270}
            />
            <span className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 text-2xl font-bold text-blue-700">
              {percent}%
            </span>
          </div>
          <span className="text-sm text-gray-500 mt-4">
            {student.creditsCompleted} / {student.totalCredits} credits completed
          </span>
        </motion.div>

        {/* Comparison Chart */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9 }}
          className="bg-white rounded-xl shadow-md p-8 border border-gray-200"
        >
          <h3 className="text-lg font-semibold mb-4 text-gray-700">
            Progression vs. Program Average
          </h3>
          <div className="h-56">
            <ResponsiveAreaBump
              data={comparisonData}
              colors={{ scheme: "category10" }}
              margin={{ top: 20, right: 80, bottom: 40, left: 80 }}
              animate={true}
              theme={{
                axis: {
                  ticks: { text: { fill: "#374151", fontSize: 13 } },
                },
                legends: {
                  text: { fill: "#374151", fontSize: 13 },
                },
                labels: {
                  text: { fill: "#374151", fontSize: 14 },
                },
              }}
            />
          </div>
          <span className="block text-sm text-gray-500 mt-6">
            Compare your pace with program average (mock data).
          </span>
        </motion.div>
      </section>
    </div>
  );
}
