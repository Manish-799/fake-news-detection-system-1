function AnalysisHistory({
  recentAnalyses,
  onClearHistory,
  onOpenHistoryItem,
  getHistoryExcerpt,
  getVerdictClass,
  formatVerdict,
  formatPercentage,
}) {
  return (
    <section className="history-card panel">
      <div className="panel-heading history-heading">
        <div>
          <span className="section-label">
            Recent Analysis
          </span>

          <h2>
            Local analysis history
          </h2>

          <p>
            Stored only in this browser using
            localStorage.
          </p>
        </div>

        {recentAnalyses.length > 0 && (
          <button
            type="button"
            className="text-button"
            onClick={onClearHistory}
          >
            Clear history
          </button>
        )}
      </div>

      {recentAnalyses.length === 0 ? (
        <div className="empty-history">
          No analyses yet. Completed predictions
          will appear here.
        </div>
      ) : (
        <div className="history-list">
          {recentAnalyses.map((item) => (
            <button
              className="history-item"
              type="button"
              key={item.id}
              onClick={() =>
                onOpenHistoryItem(item)
              }
            >
              <div className="history-copy">
                <strong>
                  {getHistoryExcerpt(item.text)}
                </strong>

                <span>
                  {new Date(
                    item.createdAt
                  ).toLocaleString()}
                </span>
              </div>

              <div className="history-result">
                <span
                  className={`history-verdict ${getVerdictClass(
                    item.verdict
                  )}`}
                >
                  {formatVerdict(
                    item.verdict
                  )}
                </span>

                <strong>
                  {formatPercentage(
                    item.probability
                  )}
                </strong>
              </div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

export default AnalysisHistory;