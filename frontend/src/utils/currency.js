// 1 USD ≈ ₹83.5 — update rate here if needed
export const USD_TO_INR = 83.5;

// Full format: ₹1,23,456
export const fmtINR = (usd) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", maximumFractionDigits: 0,
  }).format(usd * USD_TO_INR);

// Short form for chart axes: ₹12L, ₹1.2Cr
export const fmtINRShort = (usd) => {
  const inr = usd * USD_TO_INR;
  if (inr >= 1_00_00_000) return `₹${(inr / 1_00_00_000).toFixed(1)}Cr`;
  if (inr >= 1_00_000)    return `₹${(inr / 1_00_000).toFixed(1)}L`;
  return `₹${Math.round(inr).toLocaleString("en-IN")}`;
};
