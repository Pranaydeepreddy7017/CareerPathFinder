"use client";

import { motion } from "framer-motion";
import { User } from "lucide-react";
import Image from "next/image";

export default function StudentHomePage() {
  const student = {
    name: "John Doe",
  };

  const sections = [
    {
      title: "Recommended Courses",
      href: "/home/recommend",
      img: "https://cdn-icons-png.flaticon.com/512/3135/3135810.png",
    },
    {
      title: "Analytics Dashboard",
      href: "/dashboard",
      img: "https://cdn-icons-png.flaticon.com/512/1828/1828919.png",
    },
    {
      title: "Career Insights",
      href: "/home/career",
      img: "https://cdn-icons-png.flaticon.com/512/3135/3135673.png",
    },
  ];
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* 🔝 Navbar */}
      <header className="flex justify-between items-center px-10 py-4 shadow-sm bg-white sticky top-0 z-10">
        {/* Styled Logo (text only) */}
        <motion.h1
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent"
        >
          Career Pathfinder
        </motion.h1>

        {/* Nav links centered */}
        <nav className="absolute left-1/2 transform -translate-x-1/2 flex space-x-10 font-medium">
          {["Home", "Courses", "Analytics", "Career"].map((item) => (
            <motion.a
              key={item}
              href={`/${item.toLowerCase()}`}
              whileHover={{ y: -5 }} // 👆 pops up slightly
              className="relative text-gray-700 transition hover:text-blue-600"
            >
              {item}
              {/* underline animation */}
              <span className="absolute left-0 -bottom-1 w-0 h-0.5 bg-blue-600 transition-all duration-300 hover:w-full"></span>
            </motion.a>
          ))}
        </nav>

        {/* Profile dropdown */}
        <div className="relative">
          <button className="peer w-9 h-9 flex items-center justify-center rounded-full bg-gray-200 cursor-pointer">
            <User className="w-5 h-5 text-gray-600" />
          </button>
          <div className="absolute right-0 mt-3 w-44 bg-white shadow-xl rounded-xl hidden peer-hover:flex peer-focus:flex hover:flex flex-col">
            <a
              href="/student/profile"
              className="block px-5 py-2 text-sm text-gray-700 hover:bg-blue-50 hover:text-blue-600"
            >
              Profile
            </a>
            <a
              href="/settings"
              className="block px-5 py-2 text-sm text-gray-700 hover:bg-blue-50 hover:text-blue-600"
            >
              Settings
            </a>
            <a
              href="/logout"
              className="block px-5 py-2 text-sm text-gray-700 hover:bg-blue-50 hover:text-blue-600"
            >
              Logout
            </a>
          </div>
        </div>
      </header>

      {/* 🌟 Hero Section */}
      <section className="text-center py-20 bg-gradient-to-b from-white to-gray-100">
        <motion.h2
          initial={{ opacity: 0, y: 25 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl font-bold text-gray-800 mb-4"
        >
          Welcome back, {student.name}
        </motion.h2>
        <p className="text-gray-500 text-lg max-w-xl mx-auto">
          Explore curated courses, track progress, and unlock career insights
          tailored for you.
        </p>
      </section>

      {/* 🎬 Cards with icons & hover */}
      <main className="flex-1 px-10 py-12">
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-10">
          {sections.map((s) => (
            <motion.a
              key={s.title}
              href={s.href}
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 300 }}
              className="relative bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all p-8 flex flex-col justify-center items-center text-center border border-transparent hover:border-blue-400"
            >
              <Image
                src={s.img}
                alt={s.title}
                width={60}
                height={60}
                className="mb-4"
              />
              <h3 className="text-xl font-semibold text-gray-800 mb-3">
                {s.title}
              </h3>
              <p className="text-gray-500 text-sm leading-relaxed">
                Dive deeper into {s.title.toLowerCase()} crafted for your
                journey.
              </p>
            </motion.a>
          ))}
        </div>
      </main>

     

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-gray-500 border-t">
        © {new Date().getFullYear()} Student Career Pathfinder · All rights
        reserved
      </footer>
    </div>
  );
}
