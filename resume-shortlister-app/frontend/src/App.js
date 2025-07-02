import React, { useState } from "react";
import "./App.css";

const initialState = {
  degree: "",
  field: "",
  gpa: "",
  yoe: "",
  age: "",
  resume: "",
};

const degreeOptions = ["B.Tech", "M.Tech", "BSc", "MSc"];
const fieldOptions = ["Computer Science", "Electronics", "Information Technology"];

function App() {
  const [form, setForm] = useState(initialState);
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const errs = {};
    if (!form.degree) errs.degree = "Degree is required";
    if (!form.field) errs.field = "Field is required";
    if (!form.gpa || isNaN(form.gpa)) errs.gpa = "GPA must be a number";
    if (!form.yoe || isNaN(form.yoe)) errs.yoe = "YOE must be a number";
    if (!form.age || isNaN(form.age)) errs.age = "Age must be a number";
    if (!form.resume) errs.resume = "Resume is required";
    return errs;
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    const errs = validate();
    setErrors(errs);
    if (Object.keys(errs).length > 0) return;

    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/shortlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          degree: form.degree,
          field: form.field,
          gpa: parseFloat(form.gpa),
          yoe: parseFloat(form.yoe),
          age: parseInt(form.age),
          resume: form.resume,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setResult({ error: data.errors });
      } else {
        setResult(data);
      }
    } catch (err) {
      setResult({ error: "Server error" });
    }
    setLoading(false);
  };

  const handleReset = () => {
    setForm(initialState);
    setErrors({});
    setResult(null);
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Resume Shortlister</h2>
        <form onSubmit={handleSubmit} className="resume-form" autoComplete="off">
          <div className="form-row">
            <div className="input-group">
              <label>Degree</label>
              <select name="degree" value={form.degree} onChange={handleChange}>
                <option value="">Select degree</option>
                {degreeOptions.map((deg) => (
                  <option key={deg} value={deg}>{deg}</option>
                ))}
              </select>
              {errors.degree && <span className="error">{errors.degree}</span>}
            </div>
            <div className="input-group">
              <label>Field of Study</label>
              <select name="field" value={form.field} onChange={handleChange}>
                <option value="">Select field</option>
                {fieldOptions.map((f) => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
              {errors.field && <span className="error">{errors.field}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="input-group">
              <label>GPA</label>
              <input
                name="gpa"
                value={form.gpa}
                onChange={handleChange}
                placeholder="e.g. 8.5"
                type="number"
                min="0"
                max="10"
                step="0.01"
                autoFocus
              />
              {errors.gpa && <span className="error">{errors.gpa}</span>}
            </div>
            <div className="input-group">
              <label>Years of Experience</label>
              <input
                name="yoe"
                value={form.yoe}
                onChange={handleChange}
                placeholder="e.g. 2"
                type="number"
                min="0"
                max="50"
                step="0.1"
              />
              {errors.yoe && <span className="error">{errors.yoe}</span>}
            </div>
          </div>

          <div className="form-row">
            <div className="input-group">
              <label>Age</label>
              <input
                name="age"
                value={form.age}
                onChange={handleChange}
                placeholder="e.g. 25"
                type="number"
                min="18"
                max="65"
              />
              {errors.age && <span className="error">{errors.age}</span>}
            </div>
            <div className="input-group">
              {/* Empty div for grid alignment */}
            </div>
          </div>

          <div className="input-group">
            <label>Resume Text</label>
            <textarea
              name="resume"
              value={form.resume}
              onChange={handleChange}
              rows={6}
              placeholder="Paste your resume here..."
            />
            {errors.resume && <span className="error">{errors.resume}</span>}
          </div>

          <div className="form-actions">
            <button type="submit" disabled={loading}>
              {loading && <span className="loading-spinner"></span>}
              {loading ? "Analyzing..." : "Shortlist Resume"}
            </button>
            <button type="button" onClick={handleReset}>
              Reset Form
            </button>
          </div>
        </form>

        {result && (
          <div className={`result ${result.error ? 'error' : 'success'}`}>
            {result.error ? (
              <div>
                <h4>Validation Errors</h4>
                <pre>{JSON.stringify(result.error, null, 2)}</pre>
              </div>
            ) : (
              <div>
                <h4>Analysis Result</h4>
                <span className={result.shortlisted ? "shortlisted" : "not-shortlisted"}>
                  {result.shortlisted ? "Shortlisted ✅" : "Not Shortlisted ❌"}
                </span>
                <div className="probability">
                  Confidence: <b>{(result.probability * 100).toFixed(1)}%</b>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;