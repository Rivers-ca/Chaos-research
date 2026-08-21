import React, { createContext, useContext, useState, useEffect } from "react";
import { MOCK_THREADS } from "./data";
import type { Thread } from "./data";

export type UserRole = "Guest" | "Student" | "Alumni" | "Pro";
export type ThemeMode = "light" | "dark";

type AppState = { role: UserRole; savedJobIds: string[]; threads: Thread[]; theme: ThemeMode; };

interface AppContextProps extends AppState {
  cycleRole: () => void; 
  toggleSaveJob: (id: string) => void; 
  addThread: (thread: Thread) => void;
  toggleTheme: () => void;
}

const AppContext = createContext<AppContextProps | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<AppState>({ 
    role: "Student", 
    savedJobIds: [], 
    threads: MOCK_THREADS,
    theme: "light"
  });

  const cycleRole = () => {
    const roles: UserRole[] = ["Guest", "Student", "Alumni", "Pro"];
    const nextRole = roles[(roles.indexOf(state.role) + 1) % roles.length];
    setState(s => ({ ...s, role: nextRole }));
  };

  const toggleSaveJob = (id: string) => setState(s => ({ ...s, savedJobIds: s.savedJobIds.includes(id) ? s.savedJobIds.filter(j => j !== id) : [...s.savedJobIds, id] }));
  const addThread = (thread: Thread) => setState(s => ({ ...s, threads: [thread, ...s.threads] }));
  const toggleTheme = () => setState(s => ({ ...s, theme: s.theme === "light" ? "dark" : "light" }));

  return <AppContext.Provider value={{ ...state, cycleRole, toggleSaveJob, addThread, toggleTheme }}>{children}</AppContext.Provider>;
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error("useAppContext must be used within AppProvider");
  return context;
};