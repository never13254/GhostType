import CryptoKit
import Foundation
import Security

/// Stores API secrets in a local AES-256-GCM encrypted JSON file.
///
/// File location: `~/Library/Application Support/GhostType/secrets.enc`
/// Key management: A cryptographically random 256-bit key is generated on first
///                 launch and stored in the macOS Keychain with
///                 `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` (device-bound,
///                 no iCloud sync, no user prompt required at runtime).
///
/// Files previously encrypted with the legacy SHA-256(salt+UUID) key are
/// transparently re-encrypted under the new Keychain-backed key on first open.
///
/// Using a per-install random key avoids the weakness of the old scheme, where
/// the key was fully deterministic from the publicly-readable IOPlatformUUID.
final class LocalFileSecretStore: KeychainStoring {
    static let shared = LocalFileSecretStore()

    private let queue = DispatchQueue(label: "ghosttype.localfilestore.queue")
    private let fileURL: URL
    private let symmetricKey: SymmetricKey
    private let fileWriteOptions: Data.WritingOptions

    /// Keychain coordinates for the random file-encryption key.
    private static let keyService = "com.codeandchill.ghosttype.file-encryption-key"
    private static let keyAccount = "v2"

    private init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("GhostType", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes(
            [.posixPermissions: NSNumber(value: Int16(0o700))],
            ofItemAtPath: directory.path
        )
        self.fileURL = directory.appendingPathComponent("secrets.enc")
        self.fileWriteOptions = Self.defaultWriteOptions
        self.symmetricKey = Self.loadOrCreateKeychainKey()
        // self is fully initialised here; safe to call instance methods.
        migrateFromLegacyEncryptionIfNeeded()
    }

    /// Test-only initializer with explicit path and key.
    init(fileURL: URL, symmetricKey: SymmetricKey, fileWriteOptions: Data.WritingOptions = .atomic) {
        self.fileURL = fileURL
        self.symmetricKey = symmetricKey
        self.fileWriteOptions = fileWriteOptions
    }

    // MARK: - KeychainStoring

    func getSecret(forRef keyRef: String, policy: KeychainReadPolicy) throws -> String? {
        try queue.sync {
            let store = try loadStore()
            return store[keyRef]
        }
    }

    func setSecret(_ value: String, forRef keyRef: String) throws {
        try queue.sync {
            var store = (try? loadStore()) ?? [:]
            store[keyRef] = value
            try saveStore(store)
        }
    }

    func deleteSecret(forRef keyRef: String) throws {
        try queue.sync {
            var store = (try? loadStore()) ?? [:]
            store.removeValue(forKey: keyRef)
            try saveStore(store)
        }
    }

    func getSecret(for key: APISecretKey, policy: KeychainReadPolicy) throws -> String? {
        try getSecret(forRef: key.rawValue, policy: policy)
    }

    func setSecret(_ value: String, for key: APISecretKey) throws {
        try setSecret(value, forRef: key.rawValue)
    }

    func deleteSecret(for key: APISecretKey) throws {
        try deleteSecret(forRef: key.rawValue)
    }

    func deleteAllSecrets() -> KeychainRepairReport {
        queue.sync {
            try? FileManager.default.removeItem(at: fileURL)
            return makeReport(guidance: ["All saved credentials have been removed."])
        }
    }

    func runSelfCheck() -> KeychainRepairReport {
        queue.sync {
            var report = makeReport()
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                report.guidance.append("No credentials file found. Save an API key to create one.")
                return report
            }
            do {
                let store = try loadStore()
                let nonEmpty = store.values.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                report.foundCurrentItems = nonEmpty.count
                report.guidance.append("Local credentials file OK. \(nonEmpty.count) key(s) stored.")
            } catch {
                report.failures.append("Failed to read credentials file: \(error.localizedDescription)")
                report.guidance.append("Credentials file may be corrupted. Try Reset All Credentials and re-enter your keys.")
            }
            return report
        }
    }

    func runInteractiveRepair() -> KeychainRepairReport {
        runSelfCheck()
    }

    func savedSecretCount() -> Int {
        queue.sync {
            guard let store = try? loadStore() else { return 0 }
            return store.values.reduce(into: 0) { result, value in
                if !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    result += 1
                }
            }
        }
    }

    // MARK: - Encryption Internals

    /// Must be called inside `queue.sync`.
    private func loadStore() throws -> [String: String] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return [:]
        }
        let encrypted = try Data(contentsOf: fileURL)
        guard encrypted.count > 12 else {
            throw LocalFileStoreError.corruptedFile
        }
        let sealedBox = try AES.GCM.SealedBox(combined: encrypted)
        let decrypted = try AES.GCM.open(sealedBox, using: symmetricKey)
        guard let dict = try JSONSerialization.jsonObject(with: decrypted) as? [String: String] else {
            throw LocalFileStoreError.corruptedFile
        }
        return dict
    }

    /// Must be called inside `queue.sync`.
    private func saveStore(_ store: [String: String]) throws {
        let directory = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try? FileManager.default.setAttributes(
            [.posixPermissions: NSNumber(value: Int16(0o700))],
            ofItemAtPath: directory.path
        )

        let data = try JSONSerialization.data(withJSONObject: store, options: [.sortedKeys])
        let sealedBox = try AES.GCM.seal(data, using: symmetricKey)
        guard let combined = sealedBox.combined else {
            throw LocalFileStoreError.encryptionFailed
        }
        try combined.write(to: fileURL, options: fileWriteOptions)
        try? FileManager.default.setAttributes(
            [.posixPermissions: NSNumber(value: Int16(0o600))],
            ofItemAtPath: fileURL.path
        )
    }

    private static var defaultWriteOptions: Data.WritingOptions {
#if os(iOS) || os(tvOS) || os(watchOS)
        return [.atomic, .completeFileProtection]
#else
        return [.atomic]
#endif
    }

    // MARK: - Key Management

    /// Loads the random 256-bit encryption key from the Keychain, creating and
    /// persisting a new one if none is found.
    ///
    /// Storage attributes:
    /// - `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`: available after
    ///   the device has been unlocked once since boot; not backed up to iCloud;
    ///   device-bound; no interactive user prompt required.
    private static func loadOrCreateKeychainKey() -> SymmetricKey {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keyService,
            kSecAttrAccount as String: keyAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var item: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &item)

        if status == errSecSuccess, let data = item as? Data, data.count == 32 {
            return SymmetricKey(data: data)
        }

        // Generate a new cryptographically random key.
        let newKey = SymmetricKey(size: .bits256)
        let keyData = newKey.withUnsafeBytes { Data($0) }

        // Remove any malformed or outdated entry before inserting.
        if status != errSecItemNotFound {
            SecItemDelete([
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: keyService,
                kSecAttrAccount as String: keyAccount
            ] as CFDictionary)
        }

        let addAttributes: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keyService,
            kSecAttrAccount as String: keyAccount,
            kSecValueData as String: keyData,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        ]

        let addStatus = SecItemAdd(addAttributes as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            // Keychain unavailable — fall back to legacy derivation so the app
            // remains functional. This should not occur on a standard macOS install.
            return deriveLegacyKey()
        }

        return newKey
    }

    // MARK: - Migration from Legacy Key Derivation

    /// Re-encrypts the secrets file under the Keychain-backed key when it was
    /// previously written with the legacy SHA-256(salt+UUID) derived key.
    ///
    /// Called once during initialisation before `self` is exposed to other threads,
    /// so `loadStore()` / `saveStore()` are invoked directly without `queue.sync`.
    private func migrateFromLegacyEncryptionIfNeeded() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        // Fast path: current key already opens the file — nothing to migrate.
        if (try? loadStore()) != nil { return }

        // Attempt to open the file with the old derived key.
        let legacyStore = LocalFileSecretStore(fileURL: fileURL, symmetricKey: Self.deriveLegacyKey())
        guard let existingSecrets = try? legacyStore.loadStore() else { return }

        // Re-encrypt under the new Keychain-backed key (atomic write, best-effort).
        try? saveStore(existingSecrets)
    }

    // MARK: - Legacy Key (Migration Only)

    private static let legacyAppSalt = "com.codeandchill.ghosttype.local-secrets.v1"

    /// SHA-256(appSalt + IOPlatformUUID). Retained solely to decrypt files written
    /// by earlier releases. **Do not use for new encryption.**
    private static func deriveLegacyKey() -> SymmetricKey {
        let machineID = platformUUID() ?? "fallback-machine-id"
        let material = "\(legacyAppSalt).\(machineID)"
        let hash = SHA256.hash(data: Data(material.utf8))
        return SymmetricKey(data: hash)
    }

    private static func platformUUID() -> String? {
        let service = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }
        guard let uuidCF = IORegistryEntryCreateCFProperty(service, "IOPlatformUUID" as CFString, kCFAllocatorDefault, 0) else {
            return nil
        }
        return uuidCF.takeRetainedValue() as? String
    }

    private func makeReport(guidance: [String] = []) -> KeychainRepairReport {
        let bundleID = Bundle.main.bundleIdentifier ?? "unknown.bundle"
        let executablePath = Bundle.main.executableURL?.path ?? "unknown.executable"
        var report = KeychainRepairReport(
            checkedAt: Date(),
            runtimeIdentity: KeychainRuntimeIdentity(
                bundleIdentifier: bundleID,
                executablePath: executablePath,
                signingIdentifier: nil,
                teamIdentifier: nil,
                cdhash: nil,
                sandboxEnabled: false,
                keychainAccessGroups: []
            )
        )
        report.guidance = guidance
        return report
    }
}

enum LocalFileStoreError: LocalizedError {
    case corruptedFile
    case encryptionFailed

    var errorDescription: String? {
        switch self {
        case .corruptedFile:
            return "Credentials file is corrupted or unreadable."
        case .encryptionFailed:
            return "Failed to encrypt credentials."
        }
    }
}
