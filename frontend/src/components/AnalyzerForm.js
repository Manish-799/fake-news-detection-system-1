function AnalyzerForm({
  text,
  setText,
  wordCount,
  characterCount,
  error,
  loading,
  trimmedText,
  onAnalyze,
  onClear,
}) {
  return (
    <section className="analysis-card panel">
      <div className="panel-heading">
        <div>
          <span className="section-label">
            Article Analysis
          </span>

          <h2>Analyse news content</h2>

          <p>
            Paste a news headline or full article.
            Longer article text generally provides
            more linguistic features for the model.
          </p>
        </div>
      </div>

      <div className="textarea-wrapper">
        <textarea
          value={text}
          onChange={(event) =>
            setText(event.target.value)
          }
          placeholder="Paste a news headline or article here..."
          maxLength={100000}
        />

        <div className="input-stats">
          <span>
            {wordCount.toLocaleString()} words
          </span>

          <span>
            {characterCount.toLocaleString()} characters
          </span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <strong>
            Analysis unavailable
          </strong>

          <span>{error}</span>
        </div>
      )}

      <div className="action-row">
        <button
          className="secondary-button"
          type="button"
          onClick={onClear}
          disabled={loading}
        >
          Clear
        </button>

        <button
          className="primary-button"
          type="button"
          onClick={onAnalyze}
          disabled={loading || !trimmedText}
        >
          {loading
            ? "Analysing article..."
            : "Analyse Article"}
        </button>
      </div>
    </section>
  );
}

export default AnalyzerForm;