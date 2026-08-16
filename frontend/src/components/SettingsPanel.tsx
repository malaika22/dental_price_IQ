import { useEffect, useState, type FormEvent } from "react";
import {
  fetchApiKeys,
  fetchSuppliers,
  saveApiKeys,
  saveSuppliers,
  testApiKey,
} from "../api";
import { changePassword, changeUsername, getLoginUser } from "../auth";
import type { ApiKeyInfo, SupplierSource } from "../types";

interface SettingsPanelProps {
  onCredentialsChanged?: () => void;
}

type SettingsTab = "auth" | "api" | "suppliers";

const TYPE_LABELS: Record<SupplierSource["type"], string> = {
  dental_supplier: "Dental Supplier",
  marketplace: "Marketplace",
  aggregator: "Aggregator",
  other: "Other",
};

export function SettingsPanel({ onCredentialsChanged }: SettingsPanelProps) {
  const [tab, setTab] = useState<SettingsTab>("api");

  return (
    <div className="settings">
      <div className="settings-tabs" role="tablist">
        {(
          [
            ["api", "API Settings"],
            ["suppliers", "Supplier Sources"],
            ["auth", "Authentication"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            className={`settings-tabs__btn ${tab === id ? "settings-tabs__btn--on" : ""}`}
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "auth" && <AuthSettings onCredentialsChanged={onCredentialsChanged} />}
      {tab === "api" && <ApiKeySettings />}
      {tab === "suppliers" && <SupplierSettings />}
    </div>
  );
}

function AuthSettings({ onCredentialsChanged }: SettingsPanelProps) {
  const [username, setUsername] = useState(getLoginUser);
  const [currentPass, setCurrentPass] = useState("");
  const [newPass, setNewPass] = useState("");
  const [confirmPass, setConfirmPass] = useState("");
  const [newUser, setNewUser] = useState(getLoginUser);
  const [userPass, setUserPass] = useState("");
  const [passMsg, setPassMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [userMsg, setUserMsg] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  const handlePassword = (e: FormEvent) => {
    e.preventDefault();
    setPassMsg(null);
    if (newPass !== confirmPass) {
      setPassMsg({ type: "err", text: "New password and confirmation do not match." });
      return;
    }
    const err = changePassword(currentPass, newPass);
    if (err) {
      setPassMsg({ type: "err", text: err });
      return;
    }
    setCurrentPass("");
    setNewPass("");
    setConfirmPass("");
    setPassMsg({ type: "ok", text: "Password updated. Use it next time you sign in." });
    onCredentialsChanged?.();
  };

  const handleUsername = (e: FormEvent) => {
    e.preventDefault();
    setUserMsg(null);
    const err = changeUsername(userPass, newUser);
    if (err) {
      setUserMsg({ type: "err", text: err });
      return;
    }
    setUsername(getLoginUser());
    setUserPass("");
    setUserMsg({ type: "ok", text: `Login username updated to “${getLoginUser()}”.` });
    onCredentialsChanged?.();
  };

  return (
    <section className="settings-card">
      <div className="settings-card__head">
        <h2>Authentication</h2>
        <p>Update the admin login used for this portal (stored in this browser).</p>
      </div>
      <div className="settings-card__meta">
        <span className="settings-card__meta-label">Current login</span>
        <code>{username}</code>
      </div>
      <form className="settings-form" onSubmit={handlePassword}>
        <h3>Change password</h3>
        <label className="login-field">
          <span className="login-field__label">Current password</span>
          <input className="login-input" type="password" value={currentPass} onChange={(e) => setCurrentPass(e.target.value)} required />
        </label>
        <label className="login-field">
          <span className="login-field__label">New password</span>
          <input className="login-input" type="password" value={newPass} onChange={(e) => setNewPass(e.target.value)} required minLength={6} />
        </label>
        <label className="login-field">
          <span className="login-field__label">Confirm new password</span>
          <input className="login-input" type="password" value={confirmPass} onChange={(e) => setConfirmPass(e.target.value)} required minLength={6} />
        </label>
        {passMsg && <div className={`alert ${passMsg.type === "ok" ? "alert--ok" : "alert--error"}`}>{passMsg.text}</div>}
        <button type="submit" className="btn btn--primary">Save password</button>
      </form>
      <form className="settings-form settings-form--spaced" onSubmit={handleUsername}>
        <h3>Change username</h3>
        <label className="login-field">
          <span className="login-field__label">Current password</span>
          <input className="login-input" type="password" value={userPass} onChange={(e) => setUserPass(e.target.value)} required />
        </label>
        <label className="login-field">
          <span className="login-field__label">New username</span>
          <input className="login-input" type="text" value={newUser} onChange={(e) => setNewUser(e.target.value)} required minLength={3} />
        </label>
        {userMsg && <div className={`alert ${userMsg.type === "ok" ? "alert--ok" : "alert--error"}`}>{userMsg.text}</div>}
        <button type="submit" className="btn btn--soft">Save username</button>
      </form>
    </section>
  );
}

function ApiKeySettings() {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [provider, setProvider] = useState("gemini");
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [show, setShow] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testMsg, setTestMsg] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchApiKeys();
      setKeys(data.keys);
      setProvider(data.llm_provider || "gemini");
      const d: Record<string, string> = {};
      for (const k of data.keys) d[k.id] = "";
      setDrafts(d);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const save = async () => {
    setStatus(null);
    const payload: Record<string, string> = {};
    for (const [id, val] of Object.entries(drafts)) {
      if (val.trim()) payload[id] = val.trim();
    }
    try {
      const data = await saveApiKeys({ keys: payload, llm_provider: provider });
      setKeys(data.keys);
      setProvider(data.llm_provider);
      setDrafts(Object.fromEntries(data.keys.map((k) => [k.id, ""])));
      setStatus("API settings saved.");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Save failed");
    }
  };

  const runTest = async (id: string) => {
    setTesting(id);
    setTestMsg((m) => ({ ...m, [id]: "Testing…" }));
    try {
      const res = await testApiKey(id);
      setTestMsg((m) => ({ ...m, [id]: res.ok ? `✓ ${res.message}` : `✗ ${res.message}` }));
    } catch (e) {
      setTestMsg((m) => ({ ...m, [id]: e instanceof Error ? e.message : "Test failed" }));
    } finally {
      setTesting(null);
    }
  };

  if (loading) return <section className="settings-card"><p>Loading API settings…</p></section>;

  return (
    <section className="settings-card">
      <div className="settings-card__head">
        <h2>API Settings</h2>
        <p>Keys are stored in the server <code>.env</code>. Fields stay masked — paste a new key only to replace.</p>
      </div>

      <label className="login-field">
        <span className="login-field__label">LLM provider</span>
        <select className="login-input" value={provider} onChange={(e) => setProvider(e.target.value)}>
          <option value="gemini">Gemini</option>
          <option value="groq">Groq</option>
          <option value="openai">OpenAI</option>
          <option value="openrouter">OpenRouter</option>
        </select>
      </label>

      <div className="api-keys">
        {keys.map((k) => (
          <div key={k.id} className="api-key-row">
            <div className="api-key-row__head">
              <strong>{k.label}</strong>
              <span className={`api-key-row__badge ${k.configured ? "api-key-row__badge--on" : ""}`}>
                {k.configured ? `Set · ${k.masked}` : "Not configured"}
              </span>
            </div>
            <div className="api-key-row__fields">
              <input
                className="login-input"
                type={show[k.id] ? "text" : "password"}
                placeholder={k.configured ? "•••• paste new key to replace" : `Enter ${k.label} API key`}
                value={drafts[k.id] || ""}
                onChange={(e) => setDrafts((d) => ({ ...d, [k.id]: e.target.value }))}
                autoComplete="off"
              />
              <button type="button" className="btn btn--ghost" onClick={() => setShow((s) => ({ ...s, [k.id]: !s[k.id] }))}>
                {show[k.id] ? "Hide" : "Show"}
              </button>
              <button type="button" className="btn btn--soft" disabled={testing === k.id} onClick={() => void runTest(k.id)}>
                {testing === k.id ? "Testing…" : "Test connection"}
              </button>
            </div>
            {testMsg[k.id] && <p className="api-key-row__test">{testMsg[k.id]}</p>}
            <small className="api-key-row__hint">{k.env} · {k.hint}</small>
          </div>
        ))}
      </div>

      {status && <div className={`alert ${status.includes("saved") ? "alert--ok" : "alert--error"}`}>{status}</div>}
      <button type="button" className="btn btn--primary" onClick={() => void save()}>
        Save API settings
      </button>
    </section>
  );
}

function SupplierSettings() {
  const [sources, setSources] = useState<SupplierSource[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [newDomain, setNewDomain] = useState("");
  const [newType, setNewType] = useState<SupplierSource["type"]>("dental_supplier");

  const load = async () => {
    setLoading(true);
    try {
      const data = await fetchSuppliers();
      setSources(data.sources);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const persist = async (next: SupplierSource[]) => {
    setStatus(null);
    try {
      const saved = await saveSuppliers(next);
      setSources(saved);
      setStatus("Supplier sources saved.");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Save failed");
    }
  };

  const update = (domain: string, patch: Partial<SupplierSource>) => {
    setSources((prev) => prev.map((s) => (s.domain === domain ? { ...s, ...patch } : s)));
  };

  const addSource = () => {
    const domain = newDomain.trim().toLowerCase().replace(/^www\./, "");
    if (!domain || domain.includes("/") || domain.includes(" ")) {
      setStatus("Enter a valid domain like example.com");
      return;
    }
    if (sources.some((s) => s.domain === domain)) {
      setStatus("That domain is already listed.");
      return;
    }
    const row: SupplierSource = {
      id: domain,
      domain,
      label: domain,
      enabled: true,
      type: newType,
      priority: sources.length + 1,
    };
    setSources((prev) => [...prev, row]);
    setNewDomain("");
  };

  const removeSource = (domain: string) => {
    setSources((prev) => prev.filter((s) => s.domain !== domain));
  };

  if (loading) return <section className="settings-card"><p>Loading suppliers…</p></section>;

  const sorted = [...sources].sort((a, b) => a.priority - b.priority || a.domain.localeCompare(b.domain));

  return (
    <section className="settings-card">
      <div className="settings-card__head">
        <h2>Supplier Source Management</h2>
        <p>
          Enable/disable sources, set type and priority. Amazon &amp; Walmart are included as
          marketplaces for lowest public price negotiation.
        </p>
      </div>

      <div className="supplier-add">
        <input
          className="login-input"
          placeholder="new-domain.com"
          value={newDomain}
          onChange={(e) => setNewDomain(e.target.value)}
        />
        <select className="login-input" value={newType} onChange={(e) => setNewType(e.target.value as SupplierSource["type"])}>
          {Object.entries(TYPE_LABELS).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
        <button type="button" className="btn btn--soft" onClick={addSource}>Add</button>
      </div>

      <div className="supplier-table-wrap">
        <table className="supplier-table">
          <thead>
            <tr>
              <th>On</th>
              <th>Priority</th>
              <th>Domain</th>
              <th>Type</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => (
              <tr key={s.domain} className={s.enabled ? "" : "supplier-table__row--off"}>
                <td>
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    onChange={(e) => update(s.domain, { enabled: e.target.checked })}
                    aria-label={`Enable ${s.domain}`}
                  />
                </td>
                <td>
                  <input
                    className="login-input login-input--sm"
                    type="number"
                    min={1}
                    value={s.priority}
                    onChange={(e) => update(s.domain, { priority: Number(e.target.value) || 1 })}
                  />
                </td>
                <td>
                  <code>{s.domain}</code>
                  {s.label !== s.domain && <small> · {s.label}</small>}
                </td>
                <td>
                  <select
                    className="login-input"
                    value={s.type}
                    onChange={(e) => update(s.domain, { type: e.target.value as SupplierSource["type"] })}
                  >
                    {Object.entries(TYPE_LABELS).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </td>
                <td>
                  <button type="button" className="btn btn--ghost" onClick={() => removeSource(s.domain)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {status && <div className={`alert ${status.includes("saved") ? "alert--ok" : "alert--error"}`}>{status}</div>}
      <button type="button" className="btn btn--primary" onClick={() => void persist(sources)}>
        Save supplier sources
      </button>
    </section>
  );
}
