function PredictionResult({
  result,
  formatPercentage,
  formatVerdict,
  getVerdictClass,
}) {
  return (
    <section className="results-section">
      <div className="result-grid">
        <article
          className={`verdict-card panel ${getVerdictClass(
            result.verdict
          )}`}
        >
          <span className="section-label">
            Analysis Result
          </span>

          <div className="verdict-content">
            <div className="verdict-badge">
              {formatVerdict(
                result.verdict
              )}
            </div>

            <div className="probability-number">
              {formatPercentage(
                result.model_probability
              )}
            </div>

            <p>
              Model probability for the predicted class
            </p>
          </div>

          <div className="raw-prediction">
            Raw model prediction

            <strong>
              {result.model_prediction}
            </strong>
          </div>
        </article>

        <article className="probability-card panel">
          <span className="section-label">
            Class Probabilities
          </span>

          <h2>
            Prediction distribution
          </h2>

          <div className="probability-list">
            <ProbabilityBar
              label="Fake class"
              value={result.probabilities.fake}
              type="fake"
            />

            <ProbabilityBar
              label="Real class"
              value={result.probabilities.real}
              type="real"
            />
          </div>

          <p className="probability-note">
            Probabilities represent model output,
            not independently verified factual truth.
          </p>
        </article>
      </div>

      <article className="explanation-card panel">
        <div className="panel-heading">
          <div>
            <span className="section-label">
              Model Explanation
            </span>

            <h2>
              Features influencing this prediction
            </h2>

            <p>
              Active TF-IDF features are ranked by
              their signed contribution to the
              Logistic Regression decision score.
            </p>
          </div>
        </div>

        <div className="explanation-grid">
          <FeatureColumn
            title="Fake indicators"
            subtitle="Features pushing toward Fake"
            items={
              result.explanation.fake_indicators
            }
            type="fake"
          />

          <FeatureColumn
            title="Real indicators"
            subtitle="Features pushing toward Real"
            items={
              result.explanation.real_indicators
            }
            type="real"
          />
        </div>
      </article>

      <article className="stats-card panel">
        <span className="section-label">
          Article Statistics
        </span>

        <div className="stats-grid">
          <StatItem
            label="Words"
            value={
              result.article_stats.word_count
            }
          />

          <StatItem
            label="Characters"
            value={
              result.article_stats.character_count
            }
          />

          <StatItem
            label="Active model features"
            value={
              result.article_stats
                .analyzed_feature_count
            }
          />
        </div>
      </article>
    </section>
  );
}

function ProbabilityBar({
  label,
  value,
  type,
}) {
  const percentage = value * 100;

  return (
    <div className="probability-item">
      <div className="probability-header">
        <span>
          {label}
        </span>

        <strong>
          {percentage.toFixed(1)}%
        </strong>
      </div>

      <div className="probability-track">
        <div
          className={`probability-fill probability-${type}`}
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}

function FeatureColumn({
  title,
  subtitle,
  items,
  type,
}) {
  return (
    <div className="feature-column">
      <div className="feature-header">
        <div
          className={`feature-direction feature-direction-${type}`}
        />

        <div>
          <h3>
            {title}
          </h3>

          <p>
            {subtitle}
          </p>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="empty-features">
          No active features in this direction.
        </div>
      ) : (
        <div className="feature-list">
          {items.map((item) => (
            <div
              className="feature-item"
              key={`${type}-${item.term}`}
            >
              <div className="feature-term">
                <strong>
                  {item.term}
                </strong>

                <span>
                  coefficient{" "}
                  {item.coefficient.toFixed(3)}
                </span>
              </div>

              <div className="feature-contribution">
                <span>
                  contribution
                </span>

                <strong
                  className={`contribution-${type}`}
                >
                  {item.contribution > 0
                    ? "+"
                    : ""}
                  {item.contribution.toFixed(4)}
                </strong>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatItem({
  label,
  value,
}) {
  return (
    <div className="stat-item">
      <strong>
        {value.toLocaleString()}
      </strong>

      <span>
        {label}
      </span>
    </div>
  );
}

export default PredictionResult;