import { test } from 'node:test';
import assert from 'node:assert/strict';
import { handle } from './claude-review-mcp.mjs';

const call = context => ({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'review_with_claude', arguments: { context } } });
test('negotiates protocol and exposes review tool', async () => {
    assert.equal((await handle({ id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05' } })).result.protocolVersion, '2024-11-05');
    assert.equal((await handle({ id: 2, method: 'tools/list' })).result.tools[0].name, 'review_with_claude');
    assert.equal(await handle({ method: 'notifications/initialized' }), null);
});
test('rejects missing and oversized context', async () => {
    for (const context of [undefined, '', ' ', 'x'.repeat(100001)]) {
        assert.equal((await handle(call(context))).error.code, -32602);
    }
});
test('returns reviewer output and reports failure without losing availability', async () => {
    assert.equal((await handle(call('code'), async () => '검토 완료')).result.content[0].text, '검토 완료');
    assert.equal((await handle(call('code'), async () => { throw new Error('offline'); })).result.isError, true);
    assert.equal((await handle(call('code'), async () => 'retry')).result.content[0].text, 'retry');
});
test('prevents concurrent Claude calls', async () => {
    let finish;
    const pending = handle(call('first'), () => new Promise(resolve => { finish = resolve; }));
    assert.equal((await handle(call('second'))).result.isError, true);
    finish('done');
    await pending;
});
