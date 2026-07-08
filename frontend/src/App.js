import React, {
  useEffect,
  useState,
} from "react";

import "./App.css";

import AnalysisHistory
  from "./components/AnalysisHistory";

import AnalyzerForm
  from "./components/AnalyzerForm";

import ModelInformation
  from "./components/ModelInformation";

import PredictionResult
  from "./components/PredictionResult";

import {
  analyzeArticle,
  getModelInfo,
} from "./services/api";

const HISTORY_KEY = "news-analysis-history";

const MAX_HISTORY_ITEMS = 6;

function App() {
  const [text, setText] = useState("");

  const [result, setResult] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);

  const [loading, setLoading] = useState(false);

  const [
    modelInfoLoading,
    setModelInfoLoading,
  ] = useState(true);

  const [error, setError] = useState("");

  const [
    recentAnalyses,
    setRecentAnalyses,
  ] = useState([]);

  useEffect(() => {
    const loadInitialData = async () => {
      setModelInfoLoading(true);

      try {
        const data = await getModelInfo();

        setModelInfo(data);
      } catch (requestError) {
        console.error(
          "Model information error:",
          requestError
        );
      } finally {
        setModelInfoLoading(false);
      }
    };

    loadInitialData();

    try {
      const savedHistory =
        localStorage.getItem(HISTORY_KEY);

      if (savedHistory) {
        const parsedHistory =
          JSON.parse(savedHistory);

        if (Array.isArray(parsedHistory)) {
          setRecentAnalyses(parsedHistory);
        }
      }
    } catch (storageError) {
      console.error(
        "Unable to load analysis history:",
        storageError
      );

      localStorage.removeItem(HISTORY_KEY);
    }
  }, []);

  const saveAnalysis = (
    articleText,
    analysisResult
  ) => {
    const historyItem = {
      id: Date.now(),
      text: articleText,
      verdict: analysisResult.verdict,
      probability:
        analysisResult.model_probability,
      createdAt: new Date().toISOString(),
    };

    setRecentAnalyses((currentHistory) => {
      const updatedHistory = [
        historyItem,
        ...currentHistory,
      ].slice(
        0,
        MAX_HISTORY_ITEMS
      );

      localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(updatedHistory)
      );

      return updatedHistory;
    });
  };

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY);

    setRecentAnalyses([]);
  };

  const trimmedText = text.trim();

  const wordCount = trimmedText
    ? trimmedText.split(/\s+/).length
    : 0;

  const characterCount = text.length;

  const handleAnalyze = async () => {
    if (!trimmedText) {
      setError(
        "Enter a headline or news article before analysing."
      );

      return;
    }

    setLoading(true);

    setError("");

    setResult(null);

    try {
      const data = await analyzeArticle(
        trimmedText
      );

      setResult(data);

      saveAnalysis(
        trimmedText,
        data
      );
    } catch (requestError) {
      console.error(
        "Analysis request failed:",
        requestError
      );

      setError(
        requestError.message ||
          "Unable to analyse the article."
      );
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    setText("");

    setResult(null);

    setError("");
  };

  const openHistoryItem = (item) => {
    setText(item.text);

    setResult(null);

    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatVerdict = (verdict) => {
    return verdict
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(
        /\b\w/g,
        (character) =>
          character.toUpperCase()
      );
  };

  const getVerdictClass = (verdict) => {
    if (verdict === "LIKELY_REAL") {
      return "verdict-real";
    }

    if (verdict === "LIKELY_FAKE") {
      return "verdict-fake";
    }

    return "verdict-uncertain";
  };

  const getHistoryExcerpt = (
    articleText
  ) => {
    if (articleText.length <= 105) {
      return articleText;
    }

    return `${articleText.slice(
      0,
      105
    )}...`;
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-content">
          <div className="brand-row">
            <div className="brand-mark">
              NC
            </div>

            <span className="brand-name">
              NewsCred
            </span>
          </div>

          <div className="hero-copy">
            <span className="eyebrow">
              NLP Classification System
            </span>

            <h1>
              News Credibility
              <span> Analysis</span>
            </h1>

            <p>
              Analyse linguistic patterns in news
              articles using TF-IDF and an
              explainable Logistic Regression
              classifier.
            </p>
          </div>

          <div className="hero-model-status">
            <span className="status-dot" />

            {modelInfoLoading
              ? "Loading model"
              : modelInfo
                ? "Model online"
                : "Model information unavailable"}
          </div>
        </div>
      </header>

      <main className="dashboard">
        <AnalyzerForm
          text={text}
          setText={setText}
          wordCount={wordCount}
          characterCount={characterCount}
          error={error}
          loading={loading}
          trimmedText={trimmedText}
          onAnalyze={handleAnalyze}
          onClear={clearAnalysis}
        />

        {result && (
          <PredictionResult
            result={result}
            formatPercentage={
              formatPercentage
            }
            formatVerdict={formatVerdict}
            getVerdictClass={
              getVerdictClass
            }
          />
        )}

        <ModelInformation
          modelInfo={modelInfo}
          formatPercentage={
            formatPercentage
          }
        />

        <AnalysisHistory
          recentAnalyses={recentAnalyses}
          onClearHistory={clearHistory}
          onOpenHistoryItem={
            openHistoryItem
          }
          getHistoryExcerpt={
            getHistoryExcerpt
          }
          getVerdictClass={
            getVerdictClass
          }
          formatVerdict={formatVerdict}
          formatPercentage={
            formatPercentage
          }
        />

        <section className="limitation-note">
          <strong>
            Model limitation
          </strong>

          <p>
            This system performs supervised text
            classification using linguistic
            patterns learned from labelled data.
            It does not independently verify
            claims, inspect external sources, or
            perform real-time fact checking.
          </p>
        </section>
      </main>

      <footer className="footer">
        News Credibility Analysis System

        <span>
          TF-IDF · Logistic Regression · FastAPI ·
          React
        </span>
      </footer>
    </div>
  );
}

export default App;