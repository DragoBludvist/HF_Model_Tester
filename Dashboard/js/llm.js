/* llm.js — Claude API integration for detailed alert analysis */

async function requestAnalysis(alert) {
  const prompt = [
    `You are a senior SOC analyst. Analyze this security alert classified as "${alert.prediction}"`,
    `with ${(alert.confidence * 100).toFixed(1)}% confidence.`,
    `\nAlert: ${alert.alert_text}`,
    `\nRule-based note: ${alert.rule_summary.analyst_note}`,
    `\nProvide a concise analyst report (3-4 sentences): what happened,`,
    `why it matters or doesn't, and next steps. Be direct and actionable.`,
  ].join(" ");

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const data = await response.json();
  return data.content?.map((c) => c.text || "").join("") || "Analysis unavailable.";
}
