import os
import zipfile
import shutil

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
denoiser_dir = os.path.join(workspace_dir, "denoiser")
zip_filename = "Assignment-CS4814-MouhamadouAhmedNdiaye.zip"
zip_output_path = os.path.join(workspace_dir, zip_filename)

root_in_zip = "Assignment-CS4814-MouhamadouAhmedNdiaye"

print(f"[+] Creating zip archive: {zip_output_path}")

with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    # 1. Reports
    pdf_src = os.path.join(denoiser_dir, "Learned_Denoisers_Assignment_Report.pdf")
    docx_src = os.path.join(denoiser_dir, "Learned_Denoisers_Assignment_Report.docx")
    
    zf.write(pdf_src, arcname=f"{root_in_zip}/Learned_Denoisers_Report.pdf")
    zf.write(docx_src, arcname=f"{root_in_zip}/Learned_Denoisers_Report.docx")
    zf.write(os.path.join(denoiser_dir, "REPORT.md"), arcname=f"{root_in_zip}/REPORT.md")
    zf.write(os.path.join(denoiser_dir, "REPORT.html"), arcname=f"{root_in_zip}/REPORT.html")
    zf.write(os.path.join(denoiser_dir, "README.md"), arcname=f"{root_in_zip}/README.md")

    # 2. Test Image
    zf.write(os.path.join(denoiser_dir, "input_image.png"), arcname=f"{root_in_zip}/input_image.png")

    # 3. Pre-trained weights
    models_dir = os.path.join(denoiser_dir, "models")
    for f in os.listdir(models_dir):
        if f.endswith(".pth"):
            zf.write(os.path.join(models_dir, f), arcname=f"{root_in_zip}/models/{f}")

    # 4. Source package
    src_dir = os.path.join(denoiser_dir, "src")
    for f in os.listdir(src_dir):
        if f.endswith(".py"):
            zf.write(os.path.join(src_dir, f), arcname=f"{root_in_zip}/src/{f}")

    # 5. Execution scripts
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

    # 6. Outputs (Figures) - using high-quality compressed .jpg figures
    outputs_dir = os.path.join(denoiser_dir, "outputs")
    for f in os.listdir(outputs_dir):
        if f.endswith(".jpg"):
            zf.write(os.path.join(outputs_dir, f), arcname=f"{root_in_zip}/outputs/{f}")

zip_size_bytes = os.path.getsize(zip_output_path)
zip_size_mb = zip_size_bytes / (1024 * 1024)

print(f"[+] Zip successfully created!")
print(f"[+] Exact File Path: {zip_output_path}")
print(f"[+] Exact Zip File Size: {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
if zip_size_mb < 25.0:
    print(f"[+] SUCCESS: File is strictly under the 25 MB limit! Ready for school upload.")
else:
    print(f"[!] WARNING: File exceeds 25 MB limit.")
