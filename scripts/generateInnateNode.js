const fs = require('fs');
const path = require('path');
const yaml = require('yaml');

function parseSpec(specPath) {
  const data = fs.readFileSync(specPath, 'utf8');
  if (specPath.endsWith('.yaml') || specPath.endsWith('.yml')) {
    return yaml.parse(data);
  }
  return JSON.parse(data);
}

function generateNode(spec, nodeName, destDir) {
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
  }
  const pkg = {
    name: nodeName,
    version: '0.0.1',
    main: 'index.js',
    description: `Auto-generated node for ${spec.info?.title || nodeName}`,
    dependencies: {
      axios: '^1.6.7'
    }
  };
  fs.writeFileSync(path.join(destDir, 'package.json'), JSON.stringify(pkg, null, 2));
  const index = `const axios = require('axios');\n\nmodule.exports = async function execute(input) {\n  // TODO: map input to API calls\n  console.log('Running ${nodeName} node');\n  // Example GET request\n  const response = await axios.get('${spec.servers?.[0]?.url || 'http://example.com'}');\n  return response.data;\n};\n`;
  fs.writeFileSync(path.join(destDir, 'index.js'), index);
}

function main() {
  const [specPath, nodeName] = process.argv.slice(2);
  if (!specPath || !nodeName) {
    console.error('Usage: node generateInnateNode.js <openapi-spec> <node-name>');
    process.exit(1);
  }
  const spec = parseSpec(specPath);
  const destDir = path.join('generated-nodes', nodeName);
  generateNode(spec, nodeName, destDir);
  console.log(`Generated node in ${destDir}`);
}

main();
