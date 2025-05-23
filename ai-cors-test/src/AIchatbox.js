import React, { useState, useEffect } from "react";

function AIChatbox() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  // Check AI server health on component mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch("http://localhost:8000/health");
      const data = await res.json();
      setHealth(data);
      setError(null);
    } catch (err) {
      setHealth(null);
      setError("Cannot connect to AI server");
    }
  };

  const sendQuery = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    setResponse(null);
    
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
    } catch (error) {
      setError(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "800px", margin: "0 auto" }}>
      <h2>AI Agent CORS Test</h2>
      
      {/* Server Status */}
      <div style={{ 
        marginBottom: "1rem", 
        padding: "0.5rem", 
        backgroundColor: health ? "#e8f5e9" : "#ffebee",
        borderRadius: "4px"
      }}>
        <h4 style={{ margin: "0 0 0.5rem 0" }}>Server Status:</h4>
        {health ? (
          <div>
            ✅ Connected
            <br />
            Documents processed: {health.documents_processed}
          </div>
        ) : (
          <div style={{ color: "#c62828" }}>
            ❌ Not connected
          </div>
        )}
      </div>

      {/* Query Input */}
      <div style={{ marginBottom: "1rem" }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your PDFs..."
          style={{ 
            width: "100%", 
            padding: "0.5rem",
            marginBottom: "0.5rem",
            borderRadius: "4px",
            border: "1px solid #ccc"
          }}
        />
        <button 
          onClick={sendQuery} 
          disabled={loading || !health}
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: loading || !health ? "#ccc" : "#1976d2",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: loading || !health ? "not-allowed" : "pointer"
          }}
        >
          {loading ? "Processing..." : "Send Query"}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ 
          padding: "1rem", 
          backgroundColor: "#ffebee", 
          color: "#c62828",
          borderRadius: "4px",
          marginBottom: "1rem"
        }}>
          {error}
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div style={{ 
          padding: "1rem", 
          backgroundColor: "#f5f5f5", 
          borderRadius: "4px" 
        }}>
          <h4>Response:</h4>
          <p style={{ whiteSpace: "pre-wrap" }}>{response.response}</p>
          
          <div style={{ marginTop: "1rem", fontSize: "0.9em", color: "#666" }}>
            <p>Processing time: {response.processing_time.toFixed(2)}s</p>
            
            {response.sources && response.sources.length > 0 && (
              <div>
                <h4>Sources:</h4>
                <ul style={{ margin: 0 }}>
                  {response.sources.map((source, index) => (
                    <li key={index}>
                      {source.source} (Page {source.page}) - 
                      Similarity: {(source.similarity * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default AIChatbox;
