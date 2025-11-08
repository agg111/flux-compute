import { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MainDashboard from "./components/MainDashboard";
import { Toaster } from "@/components/ui/sonner";
import { Activity } from "lucide-react";

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-slate-950">
      <nav className="bg-slate-900/95 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">
                ML Workload Manager
              </span>
            </div>
          </div>
        </div>
      </nav>
      <main>{children}</main>
      <Toaster position="top-right" richColors />
    </div>
  );
};

const Home = () => {
  return (
    <Layout>
      <MainDashboard />
    </Layout>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;