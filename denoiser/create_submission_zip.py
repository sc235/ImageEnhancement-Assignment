import os
import zipfile
import shutil

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
denoiser_dir = os.path.join(workspace_dir, "denoiser")
zip_filename = "Assignment-CS4814-MouhamadouAhmedNdiaye.zip"

root_in_zip = "Assignment-CS4814-MouhamadouAhmedNdiaye"
workspace_zip = os.path.join(workspace_dir, zip_filename)
denoiser_zip = os.path.join(denoiser_dir, zip_filename)

print(f"[+] Creating submission zip archive: {workspace_zip}")

# Remove obsolete/confusing fallback files if they exist
for obsolete in ["Learned_Denoisers_Assignment_Report.docx", "Learned_Denoisers_Assignment_Report.pdf"]:
    ob_path = os.path.join(denoiser_dir, obsolete)
    if os.path.exists(ob_path):
        try:
            os.remove(ob_path)
            print(f"[+] Cleaned up obsolete temporary file: {obsolete}")
        except Exception as e:
            print(f"[-] Could not delete {obsolete}: {e}")

with zipfile.ZipFile(workspace_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # 1. Reports (Word docx, PDF, Markdown, HTML)
    pdf_path = os.path.join(denoiser_dir, "Learned_Denoisers_Report.pdf")
    docx_path = os.path.join(denoiser_dir, "Learned_Denoisers_Report.docx")
    md_path = os.path.join(denoiser_dir, "REPORT.md")
    html_path = os.path.join(denoiser_dir, "REPORT.html")
    readme_path = os.path.join(denoiser_dir, "README.md")
    
    zf.write(pdf_path, arcname=f"{root_in_zip}/Learned_Denoisers_Report.pdf")
    zf.write(docx_path, arcname=f"{root_in_zip}/Learned_Denoisers_Report.docx")
    zf.write(md_path, arcname=f"{root_in_zip}/REPORT.md")
    zf.write(html_path, arcname=f"{root_in_zip}/REPORT.html")
    zf.write(readme_path, arcname=f"{root_in_zip}/README.md")

    # 2. Evaluation target image
    zf.write(os.path.join(denoiser_dir, "input_image.png"), arcname=f"{root_in_zip}/input_image.png")

    # 3. Model weights
    models_dir = os.path.join(denoiser_dir, "models")
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pth"):
                zf.write(os.path.join(models_dir, f), arcname=f"{root_in_zip}/models/{f}")

    # 4. Source package (modular implementations)
    src_dir = os.path.join(denoiser_dir, "src")
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            if f.endswith(".py"):
                zf.write(os.path.join(src_dir, f), arcname=f"{root_in_zip}/src/{f}")

    # 5. Execution & Reproduction Scripts
    scripts = [
        "run_dncnn.py",
        "run_ffdnet.py",
        "run_noise2noise.py",
        "run_noise2void.py",
        "run_restormer.py",
        "run_diffusion.py",
        "generate_docx.py",
        "generate_pdf.py",
        "generate_html_report.py"
    ]
    for script in scripts:
        sp = os.path.join(denoiser_dir, script)
        if os.path.exists(sp):
            zf.write(sp, arcname=f"{root_in_zip}/{script}")

    # 6. Outputs / Benchmark Figures (.jpg high-quality compressed)
    outputs_dir = os.path.join(denoiser_dir, "outputs")
    if os.path.exists(outputs_dir):
        for f in os.listdir(outputs_dir):
            if f.endswith(".jpg"):
                zf.write(os.path.join(outputs_dir, f), arcname=f"{root_in_zip}/outputs/{f}")

# Copy to denoiser folder as well
shutil.copy2(workspace_zip, denoiser_zip)

zip_bytes = os.path.getsize(workspace_zip)
zip_mb = zip_bytes / (1024 * 1024)
docx_mb = os.path.getsize(docx_path) / (1024 * 1024)
pdf_mb = os.path.getsize(pdf_path) / (1024 * 1024)

print("\n" + "="*60)
print(f" SUBMISSION PACKAGE VERIFICATION")
print("="*60)
print(f" Word Report (.docx): {docx_mb:.2f} MB ({docx_path})")
print(f" PDF Report (.pdf):   {pdf_mb:.2f} MB ({pdf_path})")
print(f" Zip Archive:         {zip_mb:.2f} MB ({workspace_zip})")
print(f" Mirror Zip Archive:  {zip_mb:.2f} MB ({denoiser_zip})")
print("="*60)
if zip_mb < 25.0 and docx_mb < 25.0 and pdf_mb < 25.0:
    print(f"[SUCCESS] All files strictly below the 25.0 MB submission threshold!")
else:
    print(f"[WARNING] Files exceed threshold.")
