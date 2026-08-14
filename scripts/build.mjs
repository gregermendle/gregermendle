import { cpSync, existsSync, mkdirSync } from "node:fs";
import { execSync } from "node:child_process";

const out = "dist";
const staticItems = ["index.html", "js", "assets", "favicon.ico", "favicon.svg", "social.png"];

mkdirSync(out, { recursive: true });
execSync("npx tailwindcss -i ./styles.css -o ./dist/output.css --minify", { stdio: "inherit" });

for (const item of staticItems) {
  if (existsSync(item)) {
    cpSync(item, `${out}/${item}`, { recursive: true });
  }
}
