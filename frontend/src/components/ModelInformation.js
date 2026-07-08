function ModelInformation({
  modelInfo,
  formatPercentage,
}) {
  return (
    <section className="model-card panel">
      <div className="panel-heading model-heading">
        <div>
          <span className="section-label">
            Model Information
          </span>

          <h2>
            Classification pipeline
          </h2>

          <p>
            Evaluation metrics are loaded directly
            from the saved training artifact.
          </p>
        </div>

        <div className="model-pill">
          API v2.0
        </div>
      </div>

      {modelInfo ? (
        <div className="model-grid">
          <ModelMetric
            label="Held-out accuracy"
            value={formatPercentage(
              modelInfo.accuracy
            )}
          />

          <ModelMetric
            label="ROC-AUC"
            value={
              modelInfo.roc_auc.toFixed(4)
            }
          />

          <ModelMetric
            label="Vocabulary"
            value={
              modelInfo.vocabulary_size
                .toLocaleString()
            }
          />

          <ModelMetric
            label="Training rows"
            value={
              modelInfo.train_rows
                .toLocaleString()
            }
          />

          <ModelMetric
            label="Test rows"
            value={
              modelInfo.test_rows
                .toLocaleString()
            }
          />

          <ModelMetric
            label="Model"
            value={modelInfo.model_type}
          />
        </div>
      ) : (
        <div className="model-unavailable">
          Model metadata is currently unavailable.
          The prediction API may still be running.
        </div>
      )}

      {modelInfo && (
        <div className="pipeline-row">
          <span>
            Article
          </span>

          <span className="pipeline-arrow">
            →
          </span>

          <span>
            Text preprocessing
          </span>

          <span className="pipeline-arrow">
            →
          </span>

          <span>
            TF-IDF
          </span>

          <span className="pipeline-arrow">
            →
          </span>

          <span>
            Logistic Regression
          </span>

          <span className="pipeline-arrow">
            →
          </span>

          <span>
            Explanation
          </span>
        </div>
      )}
    </section>
  );
}

function ModelMetric({
  label,
  value,
}) {
  return (
    <div className="model-metric">
      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>
    </div>
  );
}

export default ModelInformation;