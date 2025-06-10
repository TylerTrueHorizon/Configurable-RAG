# Configurable-RAG

This repository includes utilities for creating custom **innate-in** nodes from existing REST APIs.

## Usage

1. Install dependencies:
   ```bash
   npm install
   ```

2. Generate a node from an OpenAPI specification:
   ```bash
   npm run generate path/to/openapi.json my-node-name
   ```
   The generated package will be placed in `generated-nodes/<my-node-name>`.

3. Edit the generated files as needed to comply with innate-in's custom node requirements, then publish to npm:
   ```bash
   cd generated-nodes/<my-node-name>
   npm publish
   ```

The `generateInnateNode.js` script creates a basic node that can be further customized and linked with a specific innate-in instance.
