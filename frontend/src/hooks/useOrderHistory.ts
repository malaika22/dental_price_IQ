import { useCallback, useEffect, useState } from "react";
import { fetchOrderHistory } from "../api";
import type { OrderHistoryEntry } from "../types";

export function useOrderHistory() {
  const [history, setHistory] = useState<OrderHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const rows = await fetchOrderHistory();
      setHistory(rows);
    } catch {
      /* keep last known history if server briefly unavailable */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 8_000);
    return () => window.clearInterval(id);
  }, [refresh]);

  return { history, loading, refresh };
}
