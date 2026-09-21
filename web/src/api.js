// Client for the extraction API.
//
// Every route returns the same shape, so the UI never branches on which model
// produced a graph -- which is the property that makes the side-by-side
// comparison view meaningful.

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(message, { status, detail } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, options);
  } catch (cause) {
    // A network failure and an API error need different messages: one means
    // "start the server", the other means "the server said no".
    throw new ApiError("Cannot reach the extraction API. Is the server running?", {
      status: 0,
      detail: String(cause),
    });
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Body was not JSON; the status text is the best we have.
    }
    throw new ApiError(describeStatus(response.status), {
      status: response.status,
      detail: typeof detail === "string" ? detail : JSON.stringify(detail),
    });
  }

  return response.json();
}

function describeStatus(status) {
  if (status === 503) return "That route is not configured on this server.";
  if (status === 502) return "The extraction backend failed.";
  if (status === 422) return "The request was rejected as invalid.";
  if (status === 400) return "The text could not be analysed.";
  return `Request failed (${status}).`;
}

export function health() {
  return request("/health");
}

export function metrics() {
  return request("/metrics");
}

/**
 * Extract an argument graph.
 *
 * @param {string} text        the document to analyse
 * @param {object} options
 * @param {string} options.route  claude | local | cascade
 * @param {string} [options.model] model override for the claude route
 * @param {AbortSignal} [options.signal]
 */
export function extract(text, { route = "claude", model, signal } = {}) {
  return request("/extract", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, route, model }),
    signal,
  });
}

export const EMPTY_GRAPH = { components: [], relations: [] };
