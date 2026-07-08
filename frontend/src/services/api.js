const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL ||
  "http://127.0.0.1:8000";

function extractErrorMessage(data) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg)
      .join(" ");
  }

  return "Unable to analyse the article.";
}

export async function getModelInfo() {
  const response = await fetch(
    `${API_BASE_URL}/model-info`
  );

  if (!response.ok) {
    throw new Error(
      "Unable to load model information."
    );
  }

  return response.json();
}

export async function analyzeArticle(text) {
  const response = await fetch(
    `${API_BASE_URL}/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      extractErrorMessage(data)
    );
  }

  return data;
}