import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
const origin = process.env.DEMO_URL ?? 'http://127.0.0.1:5174';
const context = await chromium.launchPersistentContext('', { channel: 'chrome', headless: true, ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] });
const report = { chrome: context.browser().version(), requests: [], warnings: [], tests: [] };
try {
  const page = context.pages()[0];
  page.on('console', m => { if (['warning', 'error'].includes(m.type())) report.warnings.push(m.text()); });
  page.on('pageerror', e => report.warnings.push(String(e)));
  await page.route('**/*', route => {
    const request = route.request(); report.requests.push({ url: request.url(), method: request.method() });
    return new URL(request.url()).origin === origin && ['GET', 'HEAD'].includes(request.method()) ? route.continue() : route.abort();
  });
  await page.goto(origin);
  report.load = await page.evaluate(() => window.qwenDemo.load());
  for (const prompt of ['What is 12 * 13? Answer with the number only.', 'Explain how canals shaped Utrecht in at least forty words.']) {
    report.tests.push(await page.evaluate(prompt => window.qwenDemo.generate(prompt), prompt));
  }
  report.success = report.load.cpuFallbackDisabled && !report.load.gpu.isFallbackAdapter && report.load.cacheTensorCount === 128
    && report.tests[0].text.trim() === '156' && report.tests[1].completionTokens >= 20
    && report.tests.every(r => r.steps.every(s => s.allStateOnGPU && s.stateTensorsRetained === 128));
  await page.evaluate(() => window.qwenDemo.unload());
  console.log(JSON.stringify({ success: report.success, loadMs: report.load.loadMs, tests: report.tests.map(r => ({ text: r.text, tokens: r.completionTokens, tps: r.tokensPerSecondAfterFirst })) }, null, 2));
} catch (error) { report.error = String(error); console.error(error); }
finally {
  await mkdir('results', { recursive: true }); await writeFile('results/browser.json', JSON.stringify(report, null, 2) + '\n'); await context.close();
}
if (!report.success) process.exitCode = 1;
