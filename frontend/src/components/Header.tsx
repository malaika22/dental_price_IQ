interface HeaderProps {
  activeTab: "analyze" | "history";
  onTabChange: (tab: "analyze" | "history") => void;
  apiConnected: boolean;
}

export function Header({ activeTab, onTabChange, apiConnected }: HeaderProps) {
  return (
    <header className="topbar">
      <div className="topbar__brand">
        <div className="topbar__logo" aria-hidden>
          <svg viewBox="0 0 40 40" fill="none">
            <rect width="40" height="40" rx="12" fill="url(#logoGrad)" />
            <path
              d="M12 26c0-5 3.5-9 8-9s8 4 8 9"
              stroke="#fff"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
            <circle cx="20" cy="15" r="3" stroke="#fff" strokeWidth="2.2" />
            <defs>
              <linearGradient id="logoGrad" x1="0" y1="0" x2="40" y2="40">
                <stop stopColor="#6366f1" />
                <stop offset="1" stopColor="#06b6d4" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div>
          <h1>Dental Price Matcher</h1>
          <p>Smart supply pricing · live pipeline</p>
        </div>
      </div>

      <div className="topbar__actions">
        <span
          className={`status-pill ${apiConnected ? "status-pill--live" : "status-pill--down"}`}
        >
          <span className="status-pill__dot" />
          {apiConnected ? "Live" : "Offline"}
        </span>

        <nav className="seg-nav" aria-label="Main">
          <button
            type="button"
            className={`seg-nav__btn ${activeTab === "analyze" ? "seg-nav__btn--on" : ""}`}
            onClick={() => onTabChange("analyze")}
          >
            Analyze
          </button>
          <button
            type="button"
            className={`seg-nav__btn ${activeTab === "history" ? "seg-nav__btn--on" : ""}`}
            onClick={() => onTabChange("history")}
          >
            History
          </button>
        </nav>
      </div>
    </header>
  );
}
