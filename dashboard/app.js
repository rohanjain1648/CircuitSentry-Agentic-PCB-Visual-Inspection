const apiBaseUrl = document.querySelector('meta[name="api-base-url"]').content;

async function loadRuns() {
  const response = await fetch(`${apiBaseUrl}/runs`);
  const data = await response.json();
  const tbody = document.getElementById("runs-body");
  tbody.innerHTML = "";
  for (const run of data.runs) {
    const row = document.createElement("tr");
    row.innerHTML = `<td><a href="#" data-run-id="${run.run_id}">${run.run_id}</a></td>` +
                     `<td>${run.seq}</td><td>${run.next_action}</td>`;
    row.querySelector("a").addEventListener("click", (e) => {
      e.preventDefault();
      loadRunDetail(run.run_id);
    });
    tbody.appendChild(row);
  }
}

async function loadRunDetail(runId) {
  const response = await fetch(`${apiBaseUrl}/runs/${runId}`);
  const data = await response.json();
  const section = document.getElementById("run-detail");
  section.hidden = false;
  document.getElementById("run-detail-title").textContent = `Run ${runId}`;
  const container = document.getElementById("run-detail-items");
  container.innerHTML = "";

  for (const item of data.items) {
    const div = document.createElement("div");
    div.className = "run-item";
    const regionsList = (item.regions || [])
      .map((r) => `<li>${r.defect_type_guess} (conf ${r.confidence}) — ${r.action}</li>`)
      .join("");
    div.innerHTML = `<h3>Seq ${item.seq} — ${item.next_action}</h3><ul>${regionsList}</ul>`;

    if (item.next_action === "flag_for_approval" && !item.approval_status) {
      const approveBtn = document.createElement("button");
      approveBtn.textContent = "Approve";
      approveBtn.addEventListener("click", () => submitApproval(runId, item.seq, "approved"));
      const rejectBtn = document.createElement("button");
      rejectBtn.textContent = "Reject";
      rejectBtn.addEventListener("click", () => submitApproval(runId, item.seq, "rejected"));
      div.appendChild(approveBtn);
      div.appendChild(rejectBtn);
    } else if (item.approval_status) {
      div.innerHTML += `<p>Approval status: ${item.approval_status}</p>`;
    }
    container.appendChild(div);
  }
}

async function submitApproval(runId, seq, decision) {
  await fetch(`${apiBaseUrl}/runs/${runId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seq, decision }),
  });
  loadRunDetail(runId);
}

loadRuns();
setInterval(loadRuns, 5000);
