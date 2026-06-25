import { useEffect, useState } from "react";
import { checkHealth, runOrderStreaming } from "./api";
import { Header } from "./components/Header";
import { HistoryPanel } from "./components/HistoryPanel";
import { ProcessingPanel } from "./components/ProcessingPanel";
import { ResultsPanel } from "./components/ResultsPanel";
import { UploadZone } from "./components/UploadZone";
import { useElapsedTimer } from "./hooks/useElapsedTimer";
import { useJobProgress } from "./hooks/useJobProgress";
import { useOrderHistory } from "./hooks/useOrderHistory";
import type { OrderRunResult } from "./types";

export default function App() {
  const [tab, setTab] = useState<"analyze" | "history">("analyze");
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [failed, setFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OrderRunResult | null>(null);
  const [apiConnected, setApiConnected] = useState(true);
  const [panelVisible, setPanelVisible] = useState(false);

  const { history, refresh } = useOrderHistory();
  const { elapsedMs, stop: stopTimer } = useElapsedTimer(processing);
  const {
    stepStatus,
    activities,
    services,
    itemProgress,
    reference,
    progressPercent,
    handleEvent,
    reset: resetProgress,
  } = useJobProgress();

  useEffect(() => {
    const ping = () => checkHealth().then(setApiConnected);
    ping();
    const id = window.setInterval(ping, 20_000);
    return () => window.clearInterval(id);
  }, []);

  const handleAnalyze = async () => {
    if (!file || processing) return;

    setProcessing(true);
    setFailed(false);
    setError(null);
    setResult(null);
    setPanelVisible(true);
    resetProgress();

    try {
      const data = await runOrderStreaming(file, handleEvent, refresh);
      stopTimer();
      setResult(data);
      await refresh();
    } catch (err) {
      stopTimer();
      setFailed(true);
      const msg = err instanceof Error ? err.message : String(err);
      const friendly =
        msg.includes("Failed to fetch") || msg.includes("NetworkError")
          ? "Could not reach the server. Make sure the backend is running on port 8000."
          : msg;

      setError(friendly);
      await refresh();
    } finally {
      setProcessing(false);
    }
  };

  const handleClear = () => {
    if (processing) return;
    setFile(null);
    setResult(null);
    setError(null);
    setFailed(false);
    setPanelVisible(false);
    resetProgress();
  };

  return (
    <div className="app">
      <div className="app__mesh" aria-hidden />
      <div className="shell">
        <Header activeTab={tab} onTabChange={setTab} apiConnected={apiConnected} />

        {tab === "analyze" ? (
          <div className="workspace">
            <section className="hero-card">
              <div className="hero-card__intro">
                <span className="hero-card__eyebrow">Henry Schein orders</span>
                <h2>Upload your PDF to find better prices</h2>
                <p>
                  We parse every line item, search public suppliers, and build three
                  negotiation-ready Excel reports — with live progress as it runs.
                </p>
              </div>

              <UploadZone
                file={file}
                onFileSelect={(f) => {
                  if (!processing) {
                    setFile(f);
                    setResult(null);
                    setError(null);
                    setFailed(false);
                    setPanelVisible(false);
                    resetProgress();
                  }
                }}
                onInvalidFile={() => setError("Please select a PDF file.")}
                disabled={processing}
              />

              <div className="toolbar">
                <button
                  type="button"
                  className="btn btn--primary btn--lg"
                  disabled={!file || processing || !apiConnected}
                  onClick={handleAnalyze}
                >
                  {processing ? (
                    <>
                      <span className="spinner" aria-hidden /> Analyzing order…
                    </>
                  ) : (
                    <>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
                        <path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                      </svg>
                      Start analysis
                    </>
                  )}
                </button>
                <button
                  type="button"
                  className="btn btn--soft"
                  disabled={!file || processing}
                  onClick={handleClear}
                >
                  Clear
                </button>
              </div>

              {!apiConnected && (
                <div className="alert alert--warn">
                  Server is offline — start the backend, then refresh this page.
                </div>
              )}
            </section>

            {panelVisible && (
              <ProcessingPanel
                processing={processing}
                failed={failed}
                done={!processing && !failed && result !== null}
                stepStatus={stepStatus}
                progressPercent={progressPercent}
                elapsedMs={elapsedMs}
                activities={activities}
                services={services}
                itemProgress={itemProgress}
                reference={reference}
              />
            )}

            {error && <div className="alert alert--error">{error}</div>}
            {result && !processing && <ResultsPanel result={result} />}
          </div>
        ) : (
          <HistoryPanel history={history} />
        )}
      </div>
    </div>
  );
}
