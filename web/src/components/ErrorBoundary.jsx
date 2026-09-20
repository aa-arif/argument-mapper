import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          width: "100%", height: "100vh",
          background: "#0D0D0D", color: "#E8E4DF",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: "'DM Sans', sans-serif",
        }}>
          <div style={{ textAlign: "center", maxWidth: 400 }}>
            <div style={{
              width: 60, height: 60, borderRadius: "50%",
              border: "2px solid #C47B7B30",
              display: "flex", alignItems: "center", justifyContent: "center",
              margin: "0 auto 16px",
            }}>
              <span style={{ fontSize: 24, color: "#C47B7B" }}>!</span>
            </div>
            <div style={{ fontSize: 16, fontWeight: 500, color: "#C47B7B", marginBottom: 8 }}>
              Something went wrong
            </div>
            <div style={{ fontSize: 13, color: "#888", lineHeight: 1.5, marginBottom: 16 }}>
              {this.state.error?.message || "An unexpected error occurred."}
            </div>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              style={{
                background: "#C47B7B20", color: "#C47B7B",
                border: "1px solid #C47B7B30", padding: "8px 20px",
                borderRadius: 6, fontSize: 12, fontWeight: 600,
                cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
              }}
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
