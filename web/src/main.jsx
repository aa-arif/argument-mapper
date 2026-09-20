import React from "react";
import ReactDOM from "react-dom/client";
import ErrorBoundary from "./components/ErrorBoundary";
import ArgumentMapper from "./ArgumentMapper";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ArgumentMapper />
    </ErrorBoundary>
  </React.StrictMode>
);
