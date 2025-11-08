import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Activity, 
  Clock, 
  DollarSign, 
  Database, 
  Cpu, 
  RefreshCw, 
  Trash2,
  TrendingUp,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2
} from "lucide-react";
import { format } from "date-fns";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const JobsDashboard = () => {
  const [jobs, setJobs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchJobs = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setIsLoading(true);
    
    try {
      const response = await axios.get(`${API}/jobs`);
      setJobs(response.data);
    } catch (error) {
      console.error("Error fetching jobs:", error);
      toast.error("Failed to fetch jobs");
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  const handleDeleteJob = async (workloadId) => {
    if (!window.confirm("Are you sure you want to delete this job?")) {
      return;
    }

    try {
      await axios.delete(`${API}/jobs/${workloadId}`);
      toast.success("Job deleted successfully");
      fetchJobs();
    } catch (error) {
      console.error("Error deleting job:", error);
      toast.error("Failed to delete job");
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      "Pending": "bg-yellow-100 text-yellow-800 border-yellow-300",
      "Analyzing": "bg-blue-100 text-blue-800 border-blue-300",
      "Optimizing": "bg-purple-100 text-purple-800 border-purple-300",
      "Running": "bg-green-100 text-green-800 border-green-300",
      "Completed": "bg-emerald-100 text-emerald-800 border-emerald-300",
      "Failed": "bg-red-100 text-red-800 border-red-300",
      "Cancelled": "bg-gray-100 text-gray-800 border-gray-300"
    };
    return colors[status] || "bg-gray-100 text-gray-800 border-gray-300";
  };

  const getStatusIcon = (status) => {
    const icons = {
      "Pending": <Clock className="w-4 h-4" />,
      "Analyzing": <Loader2 className="w-4 h-4 animate-spin" />,
      "Optimizing": <TrendingUp className="w-4 h-4" />,
      "Running": <Activity className="w-4 h-4" />,
      "Completed": <CheckCircle2 className="w-4 h-4" />,
      "Failed": <XCircle className="w-4 h-4" />,
      "Cancelled": <AlertCircle className="w-4 h-4" />
    };
    return icons[status] || <Clock className="w-4 h-4" />;
  };

  const stats = {
    total: jobs.length,
    running: jobs.filter(j => j.status === "Running" || j.status === "Analyzing" || j.status === "Optimizing").length,
    completed: jobs.filter(j => j.status === "Completed").length,
    totalBudget: jobs.reduce((sum, j) => sum + j.budget, 0)
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]" data-testid="loading-spinner">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 text-indigo-600 animate-spin mx-auto" />
          <p className="text-slate-600 font-medium">Loading jobs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8" data-testid="jobs-dashboard">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-slate-800">Jobs Dashboard</h1>
          <p className="text-slate-600 mt-2">Monitor and manage your ML workloads</p>
        </div>
        <Button
          onClick={() => fetchJobs(true)}
          disabled={refreshing}
          data-testid="refresh-jobs-button"
          className="bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 shadow-md"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="border-0 shadow-lg bg-gradient-to-br from-indigo-500 to-blue-600 text-white">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-indigo-100">Total Jobs</p>
                <p className="text-3xl font-bold mt-2" data-testid="stat-total-jobs">{stats.total}</p>
              </div>
              <Activity className="w-10 h-10 text-indigo-200" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-amber-400 to-orange-500 text-white">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-amber-100">Active Jobs</p>
                <p className="text-3xl font-bold mt-2" data-testid="stat-active-jobs">{stats.running}</p>
              </div>
              <Loader2 className="w-10 h-10 text-amber-200 animate-spin" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-emerald-500 to-green-600 text-white">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-emerald-100">Completed</p>
                <p className="text-3xl font-bold mt-2" data-testid="stat-completed-jobs">{stats.completed}</p>
              </div>
              <CheckCircle2 className="w-10 h-10 text-emerald-200" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-0 shadow-lg bg-gradient-to-br from-purple-500 to-pink-600 text-white">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-purple-100">Total Budget</p>
                <p className="text-3xl font-bold mt-2" data-testid="stat-total-budget">${stats.totalBudget.toFixed(2)}</p>
              </div>
              <DollarSign className="w-10 h-10 text-purple-200" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Jobs List */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-slate-800">Recent Jobs</h2>
        
        {jobs.length === 0 ? (
          <Card className="border-0 shadow-lg bg-white/90 backdrop-blur-sm">
            <CardContent className="py-16 text-center">
              <Activity className="w-16 h-16 text-slate-300 mx-auto mb-4" />
              <p className="text-slate-500 text-lg font-medium" data-testid="no-jobs-message">No jobs submitted yet</p>
              <p className="text-slate-400 mt-2">Submit your first ML workload to get started</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <Card 
                key={job.workload_id} 
                className="border-0 shadow-lg bg-white/90 backdrop-blur-sm card-hover"
                data-testid={`job-card-${job.workload_id}`}
              >
                <CardContent className="p-6">
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    {/* Left Section */}
                    <div className="flex-1 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-xl font-bold text-slate-800" data-testid={`job-model-name-${job.workload_id}`}>
                              {job.model_name}
                            </h3>
                            <Badge 
                              className={`${getStatusColor(job.status)} flex items-center gap-1 font-semibold border`}
                              data-testid={`job-status-${job.workload_id}`}
                            >
                              {getStatusIcon(job.status)}
                              {job.status}
                            </Badge>
                          </div>
                          <p className="text-sm text-slate-500 font-mono" data-testid={`job-workload-id-${job.workload_id}`}>
                            ID: {job.workload_id}
                          </p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="flex items-center gap-2 text-sm">
                          <Activity className="w-4 h-4 text-indigo-600" />
                          <span className="text-slate-600 font-medium" data-testid={`job-workload-type-${job.workload_id}`}>
                            {job.workload_type}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Database className="w-4 h-4 text-blue-600" />
                          <span className="text-slate-600 font-medium" data-testid={`job-datasize-${job.workload_id}`}>
                            {job.datasize}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Clock className="w-4 h-4 text-amber-600" />
                          <span className="text-slate-600 font-medium" data-testid={`job-duration-${job.workload_id}`}>
                            {job.duration}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <DollarSign className="w-4 h-4 text-green-600" />
                          <span className="text-slate-600 font-semibold" data-testid={`job-budget-${job.workload_id}`}>
                            ${job.budget.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {(job.precision || job.framework) && (
                        <div className="flex gap-2 flex-wrap">
                          {job.precision && (
                            <Badge variant="outline" className="bg-slate-50" data-testid={`job-precision-${job.workload_id}`}>
                              <Cpu className="w-3 h-3 mr-1" />
                              {job.precision}
                            </Badge>
                          )}
                          {job.framework && (
                            <Badge variant="outline" className="bg-slate-50" data-testid={`job-framework-${job.workload_id}`}>
                              {job.framework}
                            </Badge>
                          )}
                        </div>
                      )}

                      <p className="text-xs text-slate-400" data-testid={`job-created-at-${job.workload_id}`}>
                        Created {format(new Date(job.created_at), "PPp")}
                      </p>
                    </div>

                    {/* Right Section - Actions */}
                    <div className="flex lg:flex-col gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteJob(job.workload_id)}
                        data-testid={`delete-job-button-${job.workload_id}`}
                        className="border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700 hover:border-red-400"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default JobsDashboard;