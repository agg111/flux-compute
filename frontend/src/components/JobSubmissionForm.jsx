import { useState } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, Rocket, Sparkles } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const JobSubmissionForm = ({ onSuccess }) => {
  const [formData, setFormData] = useState({
    model_name: "",
    datasize: "",
    workload_type: "",
    duration: "",
    budget: "",
    precision: "",
    framework: ""
  });
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
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

      // Add optional fields if provided
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
      
      // Reset form
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
      
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      console.error("Error submitting job:", error);
      toast.error(error.response?.data?.detail || "Failed to submit job. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="shadow-xl border-0 bg-white/90 backdrop-blur-sm" data-testid="job-submission-form">
      <CardHeader className="space-y-2 pb-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-blue-600 rounded-xl flex items-center justify-center">
            <Rocket className="w-6 h-6 text-white" />
          </div>
          <div>
            <CardTitle className="text-2xl font-bold text-slate-800">Submit ML Job</CardTitle>
            <CardDescription className="text-slate-600 mt-1">
              Configure your workload and let our agents optimize resource allocation
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Required Fields */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="model_name" className="text-sm font-semibold text-slate-700">
                Model Name <span className="text-red-500">*</span>
              </Label>
              <Input
                id="model_name"
                data-testid="input-model-name"
                placeholder="e.g., llama-3-70b, gpt-4, bert-large"
                value={formData.model_name}
                onChange={(e) => handleChange("model_name", e.target.value)}
                className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="datasize" className="text-sm font-semibold text-slate-700">
                Data Size <span className="text-red-500">*</span>
              </Label>
              <Input
                id="datasize"
                data-testid="input-datasize"
                placeholder="e.g., 10GB, 500MB, 2TB"
                value={formData.datasize}
                onChange={(e) => handleChange("datasize", e.target.value)}
                className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="workload_type" className="text-sm font-semibold text-slate-700">
                Workload Type <span className="text-red-500">*</span>
              </Label>
              <Select 
                value={formData.workload_type} 
                onValueChange={(value) => handleChange("workload_type", value)}
                required
              >
                <SelectTrigger 
                  id="workload_type" 
                  data-testid="select-workload-type"
                  className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                >
                  <SelectValue placeholder="Select workload type" />
                </SelectTrigger>
                <SelectContent>
                  {workloadTypes.map((type) => (
                    <SelectItem key={type.value} value={type.value} data-testid={`workload-option-${type.value.toLowerCase().replace(/\s+/g, '-')}`}>
                      {type.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="duration" className="text-sm font-semibold text-slate-700">
                Duration <span className="text-red-500">*</span>
              </Label>
              <Input
                id="duration"
                data-testid="input-duration"
                placeholder="e.g., 2 hours, 30 minutes, 1 day"
                value={formData.duration}
                onChange={(e) => handleChange("duration", e.target.value)}
                className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="budget" className="text-sm font-semibold text-slate-700">
                Budget (USD) <span className="text-red-500">*</span>
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
                className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
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
                className="w-full flex items-center justify-between p-4 bg-slate-50 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-600" />
                  <span className="font-semibold text-slate-700">Advanced Options</span>
                </div>
                <ChevronDown
                  className={`w-5 h-5 text-slate-600 transition-transform ${isAdvancedOpen ? 'rotate-180' : ''}`}
                />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="precision" className="text-sm font-semibold text-slate-700">
                  Precision Type
                </Label>
                <Select 
                  value={formData.precision} 
                  onValueChange={(value) => handleChange("precision", value)}
                >
                  <SelectTrigger 
                    id="precision" 
                    data-testid="select-precision"
                    className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                  >
                    <SelectValue placeholder="Select precision type" />
                  </SelectTrigger>
                  <SelectContent>
                    {precisionTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value} data-testid={`precision-option-${type.value.toLowerCase().replace(/\s+/g, '-')}`}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="framework" className="text-sm font-semibold text-slate-700">
                  Framework
                </Label>
                <Select 
                  value={formData.framework} 
                  onValueChange={(value) => handleChange("framework", value)}
                >
                  <SelectTrigger 
                    id="framework" 
                    data-testid="select-framework"
                    className="border-slate-300 focus:border-indigo-500 focus:ring-indigo-500"
                  >
                    <SelectValue placeholder="Select framework" />
                  </SelectTrigger>
                  <SelectContent>
                    {frameworkTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value} data-testid={`framework-option-${type.value.toLowerCase()}`}>
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
            className="w-full h-12 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-700 hover:to-blue-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
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
  );
};

export default JobSubmissionForm;