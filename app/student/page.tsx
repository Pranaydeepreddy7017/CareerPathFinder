"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { User } from "lucide-react";

export default function StudentLoginPage() {
  const [name, setName] = useState("");
  const [schoolId, setSchoolId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      // Call backend API
      const res = await fetch("http://127.0.0.1:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, student_id: schoolId }),
      });

      if (!res.ok) {
        throw new Error("Login failed. Please check your School ID.");
      }

      const data = await res.json();

      // Save user data in localStorage
      localStorage.setItem("studentData", JSON.stringify(data));

      // Redirect to home
      router.push("/home");
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-blue-100 via-white to-purple-100 px-6">
      <div className="bg-white shadow-lg rounded-2xl p-10 w-full max-w-md border border-gray-100">
        {/* Icon & Title */}
        <div className="flex flex-col items-center mb-6">
          <div className="bg-blue-600 text-white p-4 rounded-full shadow-md">
            <User className="w-6 h-6" />
          </div>
          <h1 className="text-2xl font-bold text-gray-800 mt-3">Student Login</h1>
          <p className="text-sm text-gray-500 mt-1">
            Enter your details to continue
          </p>
        </div>

        {/* Error */}
        {error && (
          <p className="text-red-600 text-sm text-center mb-4">{error}</p>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Full Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 p-3 text-sm shadow-sm 
             text-gray-900 placeholder-gray-400 
             focus:border-blue-500 focus:ring focus:ring-blue-200"
              placeholder="John Doe"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              School ID
            </label>
            <input
              type="text"
              value={schoolId}
              onChange={(e) => setSchoolId(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 p-3 text-sm shadow-sm 
             text-gray-900 placeholder-gray-400 
             focus:border-blue-500 focus:ring focus:ring-blue-200"
              placeholder="RQ34157"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold text-sm hover:bg-blue-700 transition shadow disabled:opacity-50"
          >
            {loading ? "Logging in..." : "Continue"}
          </button>
        </form>

        {/* Footer */}
        <p className="text-xs text-gray-400 text-center mt-6">
          © {new Date().getFullYear()} Student Career Pathfinder
        </p>
      </div>
    </div>
  );
}
