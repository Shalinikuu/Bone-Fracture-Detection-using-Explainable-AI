import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
from PIL import Image
import datetime
import base64
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors

# ---------------- Load model ----------------
model_eff = tf.keras.models.load_model('fracture_model_finetuned_v2.keras')
base_model = model_eff.layers[2]

STANDARD_THRESHOLD = 0.50
CLINICAL_THRESHOLD = 0.75

# Fixed validation-set stats (from offline evaluation) - used to draw the
# static recall/precision-vs-threshold curve and the ROC curve.
THRESH_SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
RECALL_SWEEP = [0.070, 0.224, 0.357, 0.469, 0.615, 0.699, 0.755, 0.832, 0.902]
PRECISION_SWEEP = [0.909, 0.889, 0.680, 0.540, 0.506, 0.444, 0.383, 0.337, 0.264]
ROC_AUC = 0.827

FIRST_AID_FRACTURED = [
    "Do not move or put weight on the affected area.",
    "Immobilize the area using a splint or sling if possible, without forcing it into position.",
    "Apply a cold pack wrapped in cloth to reduce swelling - avoid direct ice contact with skin.",
    "Avoid eating or drinking in case sedation or surgery may be needed.",
    "Seek medical attention promptly for proper X-ray evaluation and treatment.",
]

FIRST_AID_NORMAL = [
    "No fracture pattern was detected by the model in this image.",
    "If pain, swelling, or limited movement persists, please consult a medical professional, as some hairline fractures can be difficult to detect on imaging.",
    "This tool is a screening aid only and does not replace professional radiological review.",
]

DISCLAIMER = ("This AI model is for research and educational purposes only and should not "
              "replace professional medical advice.")


def make_gradcam_heatmap(img_tensor, model_eff, base_model):
    last_conv_layer = base_model.get_layer("top_conv")
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def classify_at_threshold(confidence, threshold):
    """confidence is P(Non_fractured). A HIGHER threshold widens the 'Fractured'
    zone, which is what raises recall on the minority Fractured class."""
    is_fractured = confidence <= threshold
    label = "Fractured" if is_fractured else "Non_fractured"
    display_confidence = (1 - confidence) if is_fractured else confidence
    return label, display_confidence, is_fractured


def predict_with_gradcam(image_pil):
    image_resized = image_pil.convert('RGB').resize((224, 224))
    img_array = np.array(image_resized).astype('float32')
    img_tensor = tf.expand_dims(img_array, axis=0)

    pred = model_eff.predict(img_tensor, verbose=0)
    confidence = float(pred[0][0])

    standard_label, standard_conf, _ = classify_at_threshold(confidence, STANDARD_THRESHOLD)
    clinical_label, clinical_conf, _ = classify_at_threshold(confidence, CLINICAL_THRESHOLD)

    heatmap = make_gradcam_heatmap(img_tensor, model_eff, base_model)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    original_np = np.uint8(img_array)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)

    return {
        "overlay": overlay,
        "original_np": original_np,
        "standard_label": standard_label,
        "standard_conf": standard_conf,
        "clinical_label": clinical_label,
        "clinical_conf": clinical_conf,
        "final_label": clinical_label,
        "final_conf": clinical_conf,
    }


def build_pdf_report(result):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = f"/tmp/fracture_report_{timestamp}.pdf"

    orig_path = f"/tmp/orig_{timestamp}.png"
    overlay_path = f"/tmp/overlay_{timestamp}.png"
    Image.fromarray(result["original_np"]).save(orig_path)
    Image.fromarray(result["overlay"]).save(overlay_path)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#1f4e79"))
    c.rect(0, height - 2.2 * cm, width, 2.2 * cm, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1.5 * cm, height - 1.4 * cm, "Bone Fracture Detection Report")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(1.5 * cm, height - 2.8 * cm,
                 f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")

    label = result["final_label"]
    confidence = result["final_conf"]
    c.setFont("Helvetica-Bold", 13)
    result_color = colors.HexColor("#c0392b") if label == "Fractured" else colors.HexColor("#27ae60")
    c.setFillColor(result_color)
    c.drawString(1.5 * cm, height - 3.6 * cm, f"Prediction (clinical threshold): {label}")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 11)
    c.drawString(1.5 * cm, height - 4.2 * cm, f"Model Confidence: {confidence * 100:.1f}%")
    c.setFont("Helvetica", 9)
    c.drawString(1.5 * cm, height - 4.8 * cm,
                 f"Standard threshold (0.50) prediction: {result['standard_label']} "
                 f"({result['standard_conf']*100:.1f}%)")

    img_y = height - 13 * cm
    c.drawImage(orig_path, 1.5 * cm, img_y, width=8 * cm, height=8 * cm, preserveAspectRatio=True)
    c.drawImage(overlay_path, 10.5 * cm, img_y, width=8 * cm, height=8 * cm, preserveAspectRatio=True)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(1.5 * cm, img_y - 0.5 * cm, "Original X-ray")
    c.drawString(10.5 * cm, img_y - 0.5 * cm, "Grad-CAM Heatmap")

    text_y = img_y - 1.8 * cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(1.5 * cm, text_y, "General Safety Guidance:")
    c.setFont("Helvetica", 9)
    advice_list = FIRST_AID_FRACTURED if label == "Fractured" else FIRST_AID_NORMAL
    y = text_y - 0.6 * cm
    for line in advice_list:
        c.drawString(1.7 * cm, y, f"- {line}")
        y -= 0.55 * cm

    c.setFont("Helvetica-Oblique", 7)
    c.setFillColor(colors.grey)
    disclaimer_lines = [DISCLAIMER[i:i + 100] for i in range(0, len(DISCLAIMER), 100)]
    y = 2 * cm
    for line in disclaimer_lines:
        c.drawString(1.5 * cm, y, line)
        y -= 0.4 * cm

    c.save()
    return pdf_path


def np_to_base64(img_np):
    img = Image.fromarray(img_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_rp_svg():
    """Build a pure-SVG Recall vs Precision chart — no JS, no CDN needed."""
    W, H = 340, 180
    pad_l, pad_r, pad_t, pad_b = 40, 14, 12, 30

    cw = W - pad_l - pad_r
    ch = H - pad_t - pad_b

    def px(thresh):
        return pad_l + (thresh - 0.1) / (0.9 - 0.1) * cw

    def py(val_0_1):
        return pad_t + ch - val_0_1 * ch

    # Grid lines at 0%, 25%, 50%, 75%, 100%
    grid = ""
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = py(v)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W - pad_r}" y2="{y:.1f}" stroke="#2a2a30" stroke-width="1"/>'
        grid += f'<text x="{pad_l - 4}" y="{y + 4:.1f}" fill="#8a8a92" font-size="9" text-anchor="end">{int(v*100)}%</text>'

    # X axis tick labels
    x_labels = ""
    for t in THRESH_SWEEP:
        x = px(t)
        x_labels += f'<text x="{x:.1f}" y="{H - 4}" fill="#8a8a92" font-size="9" text-anchor="middle">{t:.1f}</text>'

    # Precision polyline (blue)
    prec_pts = " ".join(f"{px(t):.1f},{py(p/100):.1f}" for t, p in zip(THRESH_SWEEP, PRECISION_SWEEP))
    # Recall polyline (orange)
    rec_pts = " ".join(f"{px(t):.1f},{py(r/100):.1f}" for t, r in zip(THRESH_SWEEP, RECALL_SWEEP))

    # Dots
    prec_dots = "".join(f'<circle cx="{px(t):.1f}" cy="{py(p/100):.1f}" r="3" fill="#3987e5"/>' for t, p in zip(THRESH_SWEEP, PRECISION_SWEEP))
    rec_dots = "".join(f'<circle cx="{px(t):.1f}" cy="{py(r/100):.1f}" r="3" fill="#eb6834"/>' for t, r in zip(THRESH_SWEEP, RECALL_SWEEP))

    # Axis labels
    x_axis_label = f'<text x="{pad_l + cw/2:.1f}" y="{H + 2}" fill="#8a8a92" font-size="10" text-anchor="middle">Threshold</text>'

    svg = f"""<svg viewBox="0 0 {W} {H+14}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  {grid}
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ch}" stroke="#3a3a42" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t+ch}" x2="{W-pad_r}" y2="{pad_t+ch}" stroke="#3a3a42" stroke-width="1"/>
  {x_labels}
  {x_axis_label}
  <polyline points="{prec_pts}" fill="none" stroke="#3987e5" stroke-width="2" stroke-linejoin="round"/>
  <polyline points="{rec_pts}" fill="none" stroke="#eb6834" stroke-width="2" stroke-linejoin="round"/>
  {prec_dots}{rec_dots}
  <!-- Legend -->
  <rect x="{pad_l}" y="{pad_t}" width="8" height="8" rx="2" fill="#3987e5"/>
  <text x="{pad_l+11}" y="{pad_t+7}" fill="#c3c2b7" font-size="10">Precision</text>
  <rect x="{pad_l+68}" y="{pad_t}" width="8" height="8" rx="2" fill="#eb6834"/>
  <text x="{pad_l+79}" y="{pad_t+7}" fill="#c3c2b7" font-size="10">Recall</text>
</svg>"""
    return svg


def make_roc_svg():
    """Build a pure-SVG ROC curve — no JS, no CDN needed."""
    W, H = 340, 180
    pad_l, pad_r, pad_t, pad_b = 40, 14, 12, 30

    cw = W - pad_l - pad_r
    ch = H - pad_t - pad_b

    fpr = [0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.28, 0.45, 0.7, 1.0]
    tpr = [0, 0.35, 0.55, 0.68, 0.78, 0.85, 0.90, 0.95, 0.98, 1.0]

    def px(v): return pad_l + v * cw
    def py(v): return pad_t + ch - v * ch

    # Grid
    grid = ""
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        y = py(v)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r}" y2="{y:.1f}" stroke="#2a2a30" stroke-width="1"/>'
        grid += f'<text x="{pad_l-4}" y="{y+4:.1f}" fill="#8a8a92" font-size="9" text-anchor="end">{v:.2f}</text>'

    x_labels = ""
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        x = px(v)
        x_labels += f'<text x="{x:.1f}" y="{H-4}" fill="#8a8a92" font-size="9" text-anchor="middle">{v:.2f}</text>'

    # Diagonal random line
    diag = f'<line x1="{pad_l}" y1="{pad_t+ch}" x2="{W-pad_r}" y2="{pad_t}" stroke="#5f5e5a" stroke-width="1" stroke-dasharray="4 4"/>'

    # ROC fill + line
    roc_pts = " ".join(f"{px(f):.1f},{py(t):.1f}" for f, t in zip(fpr, tpr))
    fill_pts = f"{px(0):.1f},{py(0):.1f} {roc_pts} {px(1):.1f},{py(0):.1f}"

    x_axis_label = f'<text x="{pad_l + cw/2:.1f}" y="{H + 2}" fill="#8a8a92" font-size="10" text-anchor="middle">False Positive Rate</text>'
    y_axis_label = f'<text x="{8}" y="{pad_t + ch/2:.1f}" fill="#8a8a92" font-size="10" text-anchor="middle" transform="rotate(-90,8,{pad_t+ch/2:.1f})">True Positive Rate</text>'

    svg = f"""<svg viewBox="0 0 {W} {H+14}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  {grid}
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ch}" stroke="#3a3a42" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t+ch}" x2="{W-pad_r}" y2="{pad_t+ch}" stroke="#3a3a42" stroke-width="1"/>
  {x_labels}
  {x_axis_label}
  {y_axis_label}
  {diag}
  <polygon points="{fill_pts}" fill="rgba(29,158,117,0.15)"/>
  <polyline points="{roc_pts}" fill="none" stroke="#1d9e75" stroke-width="2.5" stroke-linejoin="round"/>
  <text x="{pad_l+cw-4}" y="{pad_t+16}" fill="#1d9e75" font-size="10" text-anchor="end">AUC = {ROC_AUC}</text>
</svg>"""
    return svg


def gradio_predict(input_image):
    if input_image is None:
        return "<p style='color:#aaa; padding:2rem;'>Please upload an X-ray image to begin.</p>", None

    input_image_pil = Image.fromarray(input_image).convert('RGB')
    result = predict_with_gradcam(input_image_pil)

    orig_b64 = np_to_base64(result["original_np"])
    heatmap_b64 = np_to_base64(result["overlay"])

    final_is_fracture = result["final_label"] == "Fractured"
    accent = "#e24b4a" if final_is_fracture else "#1d9e75"
    accent_bg = "rgba(226,75,74,0.12)" if final_is_fracture else "rgba(29,158,117,0.12)"
    icon = "ti-alert-triangle" if final_is_fracture else "ti-circle-check"
    final_pct = round(result["final_conf"] * 100, 1)

    std_color = "#e24b4a" if result["standard_label"] == "Fractured" else "#1d9e75"
    cli_color = "#e24b4a" if result["clinical_label"] == "Fractured" else "#1d9e75"
    agree = result["standard_label"] == result["clinical_label"]
    agree_html = (
        f"<div style='display:flex;align-items:center;gap:8px;padding:10px 14px;"
        f"background:rgba(29,158,117,0.12);border-radius:8px;color:#1d9e75;font-size:13px;font-weight:500;'>"
        f"<i class='ti ti-shield-check' style='font-size:16px;'></i>Both thresholds agree on this prediction</div>"
        if agree else
        f"<div style='display:flex;align-items:center;gap:8px;padding:10px 14px;"
        f"background:rgba(250,199,117,0.12);border-radius:8px;color:#ef9f27;font-size:13px;font-weight:500;'>"
        f"<i class='ti ti-alert-circle' style='font-size:16px;'></i>Thresholds disagree - clinical threshold changes the outcome</div>"
    )

    advice_list = FIRST_AID_FRACTURED if final_is_fracture else FIRST_AID_NORMAL
    advice_html = "".join(
        f"<div style='display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #2a2a30;'>"
        f"<i class='ti ti-point' style='font-size:8px;color:#888;margin-top:6px;flex-shrink:0;'></i>"
        f"<span style='font-size:13px;color:#c3c2b7;line-height:1.5;'>{line}</span></div>"
        for line in advice_list
    )

    rp_svg = make_rp_svg()
    roc_svg = make_roc_svg()

    html = f"""
    <div id="sr-summary" class="sr-only" style="position:absolute;width:1px;height:1px;overflow:hidden;">
      Fracture detection result: {result['final_label']} with {final_pct} percent confidence.
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css">
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:#0d0d12; padding:20px; border-radius:14px; color:#e8e8ec;">

      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:12px;">
          <div style="width:44px;height:44px;border-radius:10px;background:#1a3a6e;display:flex;align-items:center;justify-content:center;">
            <i class="ti ti-bone" style="font-size:22px;color:#7fb4f0;"></i>
          </div>
          <div>
            <div style="font-size:17px;font-weight:600;">X-Ray Fracture Detection</div>
            <div style="font-size:12px;color:#8a8a92;">AI-powered analysis with clinical thresholding</div>
          </div>
        </div>
        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:10px;padding:8px 16px;display:flex;align-items:center;gap:10px;">
          <i class="ti ti-shield-check" style="color:#1d9e75;font-size:18px;"></i>
          <div>
            <div style="font-size:11px;color:#8a8a92;">Model Confidence</div>
            <div style="font-size:16px;font-weight:600;color:#1d9e75;">{final_pct}%</div>
          </div>
        </div>
      </div>

      <div style="display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-bottom:14px;">

        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:#2a78d6;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">1</span>
            <span style="font-size:13px;font-weight:600;">Uploaded X-ray</span>
          </div>
          <img src="data:image/png;base64,{orig_b64}" style="width:100%;border-radius:8px;display:block;" />
        </div>

        <div style="background:{accent_bg};border:1px solid {accent};border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:{accent};width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">2</span>
            <span style="font-size:13px;font-weight:600;">Prediction Result</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px;margin:14px 0;">
            <i class="{icon}" style="font-size:30px;color:{accent};"></i>
            <div>
              <div style="font-size:19px;font-weight:700;color:{accent};letter-spacing:0.5px;">{result['final_label'].upper()}</div>
              <div style="font-size:11px;color:#8a8a92;">Model Confidence</div>
              <div style="font-size:22px;font-weight:700;color:{accent};">{final_pct}%</div>
            </div>
          </div>
          <div style="background:#0d0d12;border-radius:8px;height:18px;overflow:hidden;">
            <div style="background:{accent};width:{final_pct}%;height:100%;"></div>
          </div>
        </div>

        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:#7f77dd;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">3</span>
            <span style="font-size:13px;font-weight:600;">Grad-CAM Heatmap</span>
          </div>
          <img src="data:image/png;base64,{heatmap_b64}" style="width:100%;border-radius:8px;display:block;" />
        </div>
      </div>

      <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;margin-bottom:14px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <span style="background:#ef9f27;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#000;">4</span>
          <span style="font-size:13px;font-weight:600;">Threshold Comparison</span>
        </div>
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
          <tr style="color:#8a8a92;text-align:left;">
            <th style="padding:6px 0;font-weight:500;">Threshold type</th>
            <th style="padding:6px 0;font-weight:500;text-align:center;">Threshold</th>
            <th style="padding:6px 0;font-weight:500;text-align:right;">Prediction</th>
            <th style="padding:6px 0;font-weight:500;text-align:right;">Confidence</th>
          </tr>
          <tr style="border-top:1px solid #2a2a30;">
            <td style="padding:10px 0;">Standard <span style="color:#ef9f27;">(0.50)</span></td>
            <td style="padding:10px 0;text-align:center;">0.50</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;font-size:15px;color:{std_color};">{result['standard_label']}</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;font-size:15px;color:{std_color};">{round(result['standard_conf']*100,1)}%</td>
          </tr>
          <tr style="border-top:1px solid #2a2a30;">
            <td style="padding:10px 0;">Clinical <span style="color:#ef9f27;">(0.75)</span></td>
            <td style="padding:10px 0;text-align:center;">0.75</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;font-size:15px;color:{cli_color};">{result['clinical_label']}</td>
            <td style="padding:10px 0;text-align:right;font-weight:700;font-size:15px;color:{cli_color};">{round(result['clinical_conf']*100,1)}%</td>
          </tr>
        </table>
        <div style="margin-top:10px;">{agree_html}</div>
      </div>

      <div style="display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; margin-bottom:14px;">
        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:#2a78d6;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">5</span>
            <span style="font-size:13px;font-weight:600;">Recall vs Precision by Threshold</span>
          </div>
          {rp_svg}
        </div>
        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:#1d9e75;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">6</span>
            <span style="font-size:13px;font-weight:600;">ROC Curve (AUC = {ROC_AUC})</span>
          </div>
          {roc_svg}
        </div>
      </div>

      <div style="display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; margin-bottom:14px;">
        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="background:#5dcaa5;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#000;">7</span>
            <span style="font-size:13px;font-weight:600;">General Safety Guidance</span>
          </div>
          {advice_html}
        </div>

        <div style="background:#15151c;border:1px solid #2a2a30;border-radius:12px;padding:14px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
            <span style="background:#c45fe0;width:20px;height:20px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;">8</span>
            <span style="font-size:13px;font-weight:600;">Actions</span>
          </div>
          <a download="heatmap.png" href="data:image/png;base64,{heatmap_b64}"
             style="display:flex;align-items:center;gap:10px;padding:12px 14px;margin-bottom:8px;background:#2a3a5e;border-radius:8px;text-decoration:none;color:#cfe0ff;">
            <i class="ti ti-photo" style="font-size:18px;"></i>
            <div>
              <div style="font-size:13px;font-weight:600;">Download Heatmap</div>
              <div style="font-size:11px;color:#9aa8c4;">Save Grad-CAM image</div>
            </div>
          </a>
          <div style="display:flex;align-items:center;gap:10px;padding:12px 14px;background:#1c2e2a;border-radius:8px;color:#9ad9c4;">
            <i class="ti ti-file-text" style="font-size:18px;"></i>
            <div>
              <div style="font-size:13px;font-weight:600;">Download Full Report</div>
              <div style="font-size:11px;color:#7aa896;">Use the PDF panel on the left</div>
            </div>
          </div>
        </div>
      </div>

      <div style="text-align:center;font-size:11px;color:#6a6a72;padding:10px;">
        <i class="ti ti-shield-check" style="font-size:13px;vertical-align:-2px;"></i>
        {DISCLAIMER}
      </div>
    </div>

    """

    pdf_path = build_pdf_report(result)
    return html, pdf_path


custom_css = """
.gradio-container {background: #0a0a0e !important;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # X-Ray Fracture Detection
        Upload a bone X-ray image. The model predicts whether a fracture is present using a
        class-imbalance-aware clinical threshold, shows a Grad-CAM heatmap of the regions it
        focused on, and gives you a downloadable report.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload X-ray Image")
            submit_btn = gr.Button("Analyze X-ray", variant="primary")
            output_pdf = gr.File(label="Download Full Report (PDF)")
        with gr.Column(scale=2):
            output_dashboard = gr.HTML(label="Result")

    submit_btn.click(
        fn=gradio_predict,
        inputs=input_image,
        outputs=[output_dashboard, output_pdf]
    )

if __name__ == "__main__":
    demo.launch()
