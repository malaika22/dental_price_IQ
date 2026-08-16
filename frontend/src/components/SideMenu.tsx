export type AppTab = "analyze" | "history" | "settings";

interface SideMenuProps {
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
  onLogout: () => void;
}

const MENU_ITEMS: { id: AppTab; label: string; icon: "analyze" | "history" | "settings" }[] = [
  { id: "analyze", label: "Analyze", icon: "analyze" },
  { id: "history", label: "History", icon: "history" },
  { id: "settings", label: "Settings", icon: "settings" },
];

function MenuIcon({ name }: { name: "analyze" | "history" | "settings" }) {
  if (name === "analyze") {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
        <path
          d="M4 19V5M4 19h16M8 15l3-4 3 2 4-6"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  if (name === "settings") {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
        <path
          d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    );
  }

  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 6h16M4 12h16M4 18h10"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="18" cy="18" r="2.5" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

export function SideMenu({ activeTab, onTabChange, onLogout }: SideMenuProps) {
  return (
    <aside className="admin-sidebar" aria-label="Admin menu">
      <div className="admin-sidebar__brand">
        <div className="admin-sidebar__logo" aria-hidden>
          <svg viewBox="0 0 40 40" fill="none">
            <rect width="40" height="40" rx="10" fill="#0ea5e9" />
            <path
              d="M12 26c0-5 3.5-9 8-9s8 4 8 9"
              stroke="#fff"
              strokeWidth="2.2"
              strokeLinecap="round"
            />
            <circle cx="20" cy="15" r="3" stroke="#fff" strokeWidth="2.2" />
          </svg>
        </div>
        <div>
          <strong>DPM Admin</strong>
          <span>Price Matcher</span>
        </div>
      </div>

      <p className="admin-sidebar__section">Main menu</p>
      <nav className="admin-sidebar__nav">
        {MENU_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`admin-sidebar__link ${activeTab === item.id ? "admin-sidebar__link--on" : ""}`}
            onClick={() => onTabChange(item.id)}
            aria-current={activeTab === item.id ? "page" : undefined}
          >
            <MenuIcon name={item.icon} />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="admin-sidebar__footer">
        <div className="admin-sidebar__user">
          <span className="admin-sidebar__avatar" aria-hidden>
            A
          </span>
          <div>
            <strong>Admin</strong>
            <span>Administrator</span>
          </div>
        </div>
        <button type="button" className="admin-sidebar__logout" onClick={onLogout}>
          Log out
        </button>
      </div>
    </aside>
  );
}
