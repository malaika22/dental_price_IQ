import { useEffect, useState } from "react";
import { checkHealth, parseOrderPreview, runOrderStreaming } from "./api";
import { getLoginUser, isAuthenticated, logout } from "./auth";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { HistoryPanel } from "./components/HistoryPanel";
import { LoginPage } from "./components/LoginPage";
import { ParsePreviewPanel } from "./components/ParsePreviewPanel";
import { ProcessingPanel } from "./components/ProcessingPanel";
import { QuotaAlerts } from "./components/QuotaAlerts";
import { ResultsPanel } from "./components/ResultsPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import { SideMenu, type AppTab } from "./components/SideMenu";
import { UploadZone } from "./components/UploadZone";
import { useElapsedTimer } from "./hooks/useElapsedTimer";
import { useJobProgress } from "./hooks/useJobProgress";
import { useOrderHistory } from "./hooks/useOrderHistory";
import type { OrderHistoryEntry, OrderRunResult, ParsePreview } from "./types";

export default function App() {
  const [authed, setAuthed] = useState(() => isAuthenticated());
  const [tab, setTab] = useState<AppTab>("analyze");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ParsePreview | null>(null);
  const [parsing, setParsing] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [failed, setFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<OrderRunResult | null>(null);
  const [apiConnected, setApiConnected] = useState(true);
  const [panelVisible, setPanelVisible] = useState(false);
  const [adminName, setAdminName] = useState(getLoginUser);

  const { history, refresh } = useOrderHistory();
  const { elapsedMs, stop: stopTimer } = useElapsedTimer(processing);
  const {
    stepStatus,
    activities,
    services,
    itemProgress,
    reference,
    progressPercent,
    quotaAlerts,
    handleEvent,
    ingestErrorMessage,
    reset: resetProgress,
  } = useJobProgress();

  useEffect(() => {
    const ping = () => checkHealth().then(setApiConnected);
    ping();
    const id = window.setInterval(ping, 20_000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!panelVisible) return;
    const t = window.setTimeout(() => {
      document.getElementById("process-panel")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 80);
    return () => window.clearTimeout(t);
  }, [panelVisible]);

  const handlePreview = async () => {
    if (!file || parsing || processing) return;
    setParsing(true);
    setError(null);
    setResult(null);
    setFailed(false);
    setPanelVisible(false);
    setPreview(null);
    try {
      const data = await parseOrderPreview(file);
      setPreview(data);
      window.setTimeout(() => {
        document.getElementById("parse-preview")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 60);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setParsing(false);
    }
  };

  const handleConfirmSearch = async () => {
    if (!file || processing) return;

    setProcessing(true);
    setFailed(false);
    setError(null);
    setResult(null);
    setPreview(null);
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
          ? import.meta.env.VITE_API_BASE
            ? "Could not reach the API server. It may be starting up — wait a moment and try again."
            : "Could not reach the server. Make sure the backend is running on port 8000."
          : msg;

      setError(friendly);
      ingestErrorMessage(msg);
      await refresh();
    } finally {
      setProcessing(false);
    }
  };

  const handleClear = () => {
    if (processing || parsing) return;
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setFailed(false);
    setPanelVisible(false);
    resetProgress();
  };

  const handleLogout = () => {
    logout();
    setAuthed(false);
    setTab("analyze");
    handleClear();
  };

  const handleRerun = (entry: OrderHistoryEntry) => {
    setTab("analyze");
    setError(
      `To re-run “${entry.reference || entry.fileName}”, upload the original PDF again, preview line items, then confirm the price search.`,
    );
    setPreview(null);
    setResult(null);
    setPanelVisible(false);
  };

  if (!authed) {
    return <LoginPage onSuccess={() => setAuthed(true)} />;
  }

  return (
    <div className="app app--portal">
      <SideMenu activeTab={tab} onTabChange={setTab} onLogout={handleLogout} />

      <div className="portal-body">
        <Header activeTab={tab} apiConnected={apiConnected} adminName={adminName} />

        <main className="portal-main">
          {tab === "analyze" ? (
            <div className="workspace">
              <section className="hero-card">
                <div className="hero-card__intro">
                  <span className="hero-card__eyebrow">Orders</span>
                  <h2>Upload PDF for price matching</h2>
                  <p>
                    First preview parsed line items, then confirm to run the full supplier
                    search and Excel reports.
                  </p>
                </div>

                <UploadZone
                  file={file}
                  onFileSelect={(f) => {
                    if (!processing && !parsing) {
                      setFile(f);
                      setPreview(null);
                      setResult(null);
                      setError(null);
                      setFailed(false);
                      setPanelVisible(false);
                      resetProgress();
                    }
                  }}
                  onInvalidFile={() => setError("Please select a PDF file.")}
                  disabled={processing || parsing}
                />

                <div className="toolbar">
                  <button
                    type="button"
                    className="btn btn--primary btn--lg"
                    disabled={!file || processing || parsing || !apiConnected}
                    onClick={() => void handlePreview()}
                  >
                    {parsing ? (
                      <>
                        <span className="spinner" aria-hidden /> Parsing PDF…
                      </>
                    ) : (
                      "Preview line items"
                    )}
                  </button>
                  <button
                    type="button"
                    className="btn btn--soft"
                    disabled={!file || processing || parsing}
                    onClick={handleClear}
                  >
                    Clear
                  </button>
                </div>

                {!apiConnected && (
                  <div className="alert alert--warn">
                    {import.meta.env.VITE_API_BASE
                      ? "API server is offline — it may be waking up on Render. Refresh in a minute."
                      : "Server is offline — start the backend, then refresh this page."}
                  </div>
                )}
              </section>

              {preview && !processing && (
                <div id="parse-preview">
                  <ParsePreviewPanel
                    preview={preview}
                    confirming={processing}
                    onConfirm={() => void handleConfirmSearch()}
                    onCancel={() => setPreview(null)}
                  />
                </div>
              )}

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
                  quotaAlerts={quotaAlerts}
                />
              )}

              {quotaAlerts.length > 0 && !panelVisible && <QuotaAlerts alerts={quotaAlerts} />}

              {error && <div className="alert alert--error">{error}</div>}
              {result && !processing && <ResultsPanel result={result} />}
            </div>
          ) : tab === "history" ? (
            <HistoryPanel history={history} onRerun={handleRerun} />
          ) : (
            <SettingsPanel onCredentialsChanged={() => setAdminName(getLoginUser())} />
          )}
        </main>

        <Footer />
      </div>
    </div>
  );
}
