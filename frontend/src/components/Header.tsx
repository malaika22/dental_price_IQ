import type { AppTab } from "./SideMenu";

interface HeaderProps {
  activeTab: AppTab;
  apiConnected: boolean;
  adminName?: string;
}

const PAGE_META: Record<AppTab, { title: string; crumb: string }> = {
  analyze: { title: "Analyze Orders", crumb: "Analyze" },
  history: { title: "Order History", crumb: "History" },
  settings: { title: "Admin Settings", crumb: "Settings" },
};

export function Header({ activeTab, apiConnected, adminName = "admin" }: HeaderProps) {
  const page = PAGE_META[activeTab];

  return (
    <header className="admin-topbar">
      <div className="admin-topbar__left">
        <nav className="admin-breadcrumb" aria-label="Breadcrumb">
          <span>Admin</span>
          <span className="admin-breadcrumb__sep">/</span>
          <span className="admin-breadcrumb__current">{page.crumb}</span>
        </nav>
        <h1 className="admin-topbar__title">{page.title}</h1>
      </div>

      <div className="admin-topbar__right">
        <span
          className={`status-pill ${apiConnected ? "status-pill--live" : "status-pill--down"}`}
        >
          <span className="status-pill__dot" />
          API {apiConnected ? "Online" : "Offline"}
        </span>
        <div className="admin-topbar__chip">
          <span className="admin-topbar__chip-avatar" aria-hidden>
            {(adminName[0] || "A").toUpperCase()}
          </span>
          <span>{adminName}</span>
        </div>
      </div>
    </header>
  );
}
