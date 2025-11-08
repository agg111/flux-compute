import { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import JobSubmissionForm from "./components/JobSubmissionForm";
import JobsDashboard from "./components/JobsDashboard";
import { Toaster } from "@/components/ui/sonner";
import { Activity } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Layout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <nav className="bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-lg flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent">
                ML Workload Manager
              </span>
            </Link>
            <div className="flex gap-4">
              <Link
                to="/"
                data-testid="nav-dashboard-link"
                className="px-4 py-2 text-sm font-medium text-slate-700 hover:text-indigo-600 transition-colors"
              >
                Dashboard
              </Link>
              <Link
                to="/submit"
                data-testid="nav-submit-link"
                className="px-4 py-2 text-sm font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors shadow-md hover:shadow-lg"
              >
                Submit Job
              </Link>
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
      <JobsDashboard />
    </Layout>
  );
};

const SubmitJob = () => {
  const navigate = useNavigate();
  
  const handleSuccess = () => {
    navigate("/");
  };
  
  return (
    <Layout>
      <div className="max-w-3xl mx-auto px-4 py-8">
        <JobSubmissionForm onSuccess={handleSuccess} />
      </div>
    </Layout>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/submit" element={<SubmitJob />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;