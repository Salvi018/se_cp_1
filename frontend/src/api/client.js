import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 30000,  // 30s — Render free tier cold starts can be slow
});

// Unwrap { success, data } envelope
api.interceptors.response.use(
  (res) => {
    if (res.data?.success === true && res.data?.data !== undefined) {
      return { ...res, data: res.data.data };
    }
    return res;
  },
  (err) => {
    const msg =
      err.response?.data?.details?.[0] ||
      err.response?.data?.error ||
      err.response?.data?.message ||
      (err.code === "ECONNABORTED"
        ? "Request timed out. Backend may be cold-starting, try again in 30s."
        : null) ||
      "Something went wrong.";
    return Promise.reject(new Error(msg));
  }
);

export const postPredict          = (data) => api.post("/predict", data).then((r) => r.data);
export const getHistory           = (page) => api.get(`/history?page=${page}&per_page=10`).then((r) => r.data);
export const getMetrics           = ()     => api.get("/models/metrics").then((r) => r.data);
export const getFeatureImportance = ()     => api.get("/charts/feature-importance").then((r) => r.data);
export const getPredictedVsActual = ()     => api.get("/charts/predicted-vs-actual").then((r) => r.data);
