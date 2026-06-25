import { useEffect, useRef, useState } from "react";

export function useElapsedTimer(running: boolean) {
  const [elapsedMs, setElapsedMs] = useState(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (!running) return;

    startRef.current = Date.now();
    setElapsedMs(0);

    const id = window.setInterval(() => {
      if (startRef.current) {
        setElapsedMs(Date.now() - startRef.current);
      }
    }, 250);

    return () => window.clearInterval(id);
  }, [running]);

  const stop = () => {
    if (startRef.current) {
      const final = Date.now() - startRef.current;
      setElapsedMs(final);
      return final;
    }
    return elapsedMs;
  };

  return { elapsedMs, stop };
}
