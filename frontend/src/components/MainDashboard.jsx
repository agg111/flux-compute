import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  Activity, 
  Clock, 
  DollarSign, 
  Database, 
  Cpu, 
  RefreshCw, 
  Trash2,
  Rocket,
  ChevronDown,
  Sparkles,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2
} from "lucide-react";
import { format } from "date-fns";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const MainDashboard = () => {
  const [jobs, setJobs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [formData, setFormData] = useState({
    model_name: "",
    datasize: "",
    workload_type: "",
    duration: "",
    budget: "",
    precision: "",
    framework: ""
  });

  const workloadTypes = [
    { value: "Inference", label: "Inference" },
    { value: "Fine-tuning", label: "Fine-tuning" },
    { value: "Embeddings Generation", label: "Embeddings Generation" },
    { value: "Training", label: "Training" }
  ];

  const precisionTypes = [
    { value: "FP32", label: "FP32 (Full Precision)" },
    { value: "FP16", label: "FP16 (Half Precision)" },
    { value: "INT8", label: "INT8 (Quantized)" },
    { value: "Mixed Precision", label: "Mixed Precision" }
  ];

  const frameworkTypes = [
    { value: "PyTorch", label: "PyTorch" },
    { value: "TensorFlow", label: "TensorFlow" },
    { value: "JAX", label: "JAX" },
    { value: "ONNX", label: "ONNX" }
  ];

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

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.model_name || !formData.datasize || !formData.workload_type || 
        !formData.duration || !formData.budget) {
      toast.error("Please fill in all required fields");
      return;
    }

    setIsSubmitting(true);

    try {
      const submitData = {
        model_name: formData.model_name,
        datasize: formData.datasize,
        workload_type: formData.workload_type,
        duration: formData.duration,
        budget: parseFloat(formData.budget)
      };

      if (formData.precision) {
        submitData.precision = formData.precision;
      }
      if (formData.framework) {
        submitData.framework = formData.framework;
      }

      const response = await axios.post(`${API}/jobs`, submitData);
      
      toast.success(
        `Job submitted successfully! Workload ID: ${response.data.workload_id.substring(0, 8)}...`,
        { duration: 5000 }
      );
      
      setFormData({
        model_name: "",
        datasize: "",
        workload_type: "",
        duration: "",
        budget: "",
        precision: "",
        framework: ""
      });
      setIsAdvancedOpen(false);
      
      fetchJobs();
    } catch (error) {
      console.error("Error submitting job:", error);
      toast.error(error.response?.data?.detail || "Failed to submit job. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

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
      "Pending": "bg-slate-700 text-slate-200 border-slate-600",
      "Analyzing": "bg-blue-900/50 text-blue-200 border-blue-700",
      "Optimizing": "bg-blue-900/50 text-blue-200 border-blue-700",
      "Running": "bg-blue-900/50 text-blue-200 border-blue-700",
      "Completed": "bg-slate-700 text-slate-200 border-slate-600",
      "Failed": "bg-slate-700 text-slate-200 border-slate-600",
      "Cancelled": "bg-slate-700 text-slate-200 border-slate-600"
    };
    return colors[status] || "bg-slate-700 text-slate-200 border-slate-600";
  };

  const getStatusIcon = (status) => {
    const icons = {
      "Pending": <Clock className="w-4 h-4" />,
      "Analyzing": <Loader2 className="w-4 h-4 animate-spin" />,
      "Optimizing": <Activity className="w-4 h-4" />,
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
          <Loader2 className="w-12 h-12 text-blue-500 animate-spin mx-auto" />
          <p className="text-slate-400 font-medium">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8" data-testid="main-dashboard">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400 mt-2">Submit and monitor ML workloads</p>
        </div>
        <Button
          onClick={() => fetchJobs(true)}
          disabled={refreshing}
          data-testid="refresh-jobs-button"
          className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Submit Job Form - First Card */}
      <Card className="border border-slate-800 bg-slate-900" data-testid="job-submission-form">
        <CardHeader className="space-y-2 pb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
              <Rocket className="w-6 h-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-2xl font-bold text-white">Submit New Job</CardTitle>
              <CardDescription className="text-slate-400 mt-1">
                Configure your ML workload for resource optimization
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="model_name" className="text-sm font-semibold text-slate-300">
                  Model Name <span className="text-blue-400">*</span>
                </Label>
                <Input
                  id="model_name"
                  data-testid="input-model-name"
                  placeholder="e.g., llama-3-70b, gpt-4"
                  value={formData.model_name}
                  onChange={(e) => handleChange("model_name", e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus:border-blue-600"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="datasize" className="text-sm font-semibold text-slate-300">
                  Data Size <span className="text-blue-400">*</span>
                </Label>
                <Input
                  id="datasize"
                  data-testid="input-datasize"
                  placeholder="e.g., 10GB, 500MB"
                  value={formData.datasize}
                  onChange={(e) => handleChange("datasize", e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus:border-blue-600"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="workload_type" className="text-sm font-semibold text-slate-300">
                  Workload Type <span className="text-blue-400">*</span>
                </Label>
                <Select 
                  value={formData.workload_type} 
                  onValueChange={(value) => handleChange("workload_type", value)}
                  required
                >
                  <SelectTrigger 
                    id="workload_type" 
                    data-testid="select-workload-type"
                    className="bg-slate-800 border-slate-700 text-white focus:border-blue-600"
                  >
                    <SelectValue placeholder="Select workload type" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700">
                    {workloadTypes.map((type) => (
                      <SelectItem 
                        key={type.value} 
                        value={type.value} 
                        data-testid={`workload-option-${type.value.toLowerCase().replace(/\s+/g, '-')}`}
                        className="text-white focus:bg-slate-700 focus:text-white"
                      >
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="duration" className="text-sm font-semibold text-slate-300">
                  Duration <span className="text-blue-400">*</span>
                </Label>
                <Input
                  id="duration"
                  data-testid="input-duration"
                  placeholder="e.g., 2 hours, 1 day"
                  value={formData.duration}
                  onChange={(e) => handleChange("duration", e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus:border-blue-600"
                  required
                />
              </div>

              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="budget" className="text-sm font-semibold text-slate-300">
                  Budget (USD) <span className="text-blue-400">*</span>
                </Label>
                <Input
                  id="budget"
                  data-testid="input-budget"
                  type="number"
                  step="0.01"
                  min="0"
                  placeholder="e.g., 100.00"
                  value={formData.budget}
                  onChange={(e) => handleChange("budget", e.target.value)}
                  className="bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus:border-blue-600"
                  required
                />
              </div>
            </div>

            {/* Advanced Options */}
            <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  data-testid="toggle-advanced-options"
                  className="w-full flex items-center justify-between p-4 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg"
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-blue-400" />
                    <span className="font-semibold">Advanced Options</span>
                  </div>
                  <ChevronDown
                    className={`w-5 h-5 transition-transform ${isAdvancedOpen ? 'rotate-180' : ''}`}
                  />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="precision" className="text-sm font-semibold text-slate-300">
                    Precision Type
                  </Label>
                  <Select 
                    value={formData.precision} 
                    onValueChange={(value) => handleChange("precision", value)}
                  >
                    <SelectTrigger 
                      id="precision" 
                      data-testid="select-precision"
                      className="bg-slate-800 border-slate-700 text-white focus:border-blue-600"
                    >
                      <SelectValue placeholder="Select precision" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      {precisionTypes.map((type) => (
                        <SelectItem 
                          key={type.value} 
                          value={type.value} 
                          data-testid={`precision-option-${type.value.toLowerCase().replace(/\s+/g, '-')}`}
                          className="text-white focus:bg-slate-700 focus:text-white"
                        >
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="framework" className="text-sm font-semibold text-slate-300">
                    Framework
                  </Label>
                  <Select 
                    value={formData.framework} 
                    onValueChange={(value) => handleChange("framework", value)}
                  >
                    <SelectTrigger 
                      id="framework" 
                      data-testid="select-framework"
                      className="bg-slate-800 border-slate-700 text-white focus:border-blue-600"
                    >
                      <SelectValue placeholder="Select framework" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-700">
                      {frameworkTypes.map((type) => (
                        <SelectItem 
                          key={type.value} 
                          value={type.value} 
                          data-testid={`framework-option-${type.value.toLowerCase()}`}
                          className="text-white focus:bg-slate-700 focus:text-white"
                        >
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </CollapsibleContent>
            </Collapsible>

            <Button
              type="submit"
              data-testid="submit-job-button"
              disabled={isSubmitting}
              className="w-full h-12 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg"
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Submitting...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Rocket className="w-5 h-5" />
                  Submit Job
                </span>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Jobs List */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold text-white">Recent Jobs</h2>
        
        {jobs.length === 0 ? (
          <Card className="border border-slate-800 bg-slate-900">
            <CardContent className="py-16 text-center">
              <Activity className="w-16 h-16 text-slate-700 mx-auto mb-4" />
              <p className="text-slate-400 text-lg font-medium" data-testid="no-jobs-message">No jobs submitted yet</p>
              <p className="text-slate-500 mt-2">Submit your first ML workload above</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <Card 
                key={job.workload_id} 
                className="border border-slate-800 bg-slate-900 card-hover"
                data-testid={`job-card-${job.workload_id}`}
              >
                <CardContent className="p-6">
                  <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                    <div className="flex-1 space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-xl font-bold text-white" data-testid={`job-model-name-${job.workload_id}`}>
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
                          <Activity className="w-4 h-4 text-blue-400" />
                          <span className="text-slate-300 font-medium" data-testid={`job-workload-type-${job.workload_id}`}>
                            {job.workload_type}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Database className="w-4 h-4 text-slate-400" />
                          <span className="text-slate-300 font-medium" data-testid={`job-datasize-${job.workload_id}`}>
                            {job.datasize}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <Clock className="w-4 h-4 text-slate-400" />
                          <span className="text-slate-300 font-medium" data-testid={`job-duration-${job.workload_id}`}>
                            {job.duration}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-sm">
                          <DollarSign className="w-4 h-4 text-slate-400" />
                          <span className="text-slate-300 font-semibold" data-testid={`job-budget-${job.workload_id}`}>
                            ${job.budget.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {(job.precision || job.framework) && (
                        <div className="flex gap-2 flex-wrap">
                          {job.precision && (
                            <Badge variant="outline" className="bg-slate-800 border-slate-700 text-slate-300" data-testid={`job-precision-${job.workload_id}`}>
                              <Cpu className="w-3 h-3 mr-1" />
                              {job.precision}
                            </Badge>
                          )}
                          {job.framework && (
                            <Badge variant="outline" className="bg-slate-800 border-slate-700 text-slate-300" data-testid={`job-framework-${job.workload_id}`}>
                              {job.framework}
                            </Badge>
                          )}
                        </div>
                      )}

                      <p className="text-xs text-slate-500" data-testid={`job-created-at-${job.workload_id}`}>
                        Created {format(new Date(job.created_at), "PPp")}
                      </p>
                    </div>

                    <div className="flex lg:flex-col gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteJob(job.workload_id)}
                        data-testid={`delete-job-button-${job.workload_id}`}
                        className="border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
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

export default MainDashboard;