import React from "react";

export function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded-full">
      {children}
    </span>
  );
}
